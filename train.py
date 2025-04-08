import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from dataset import VOCDataset
from model import FractalNetDetection
from utils import (
    AverageMeter, save_checkpoint, EarlyStopping, calculate_metrics, plot_predictions, xywh2xyxy, plot_confusion_matrix
)

def parse_args():
    parser = argparse.ArgumentParser(description='FractalNet Object Detection Training')
    parser.add_argument('--config', type=str, default='config.py', help='Path to the config file')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--img-size', type=int, default=416, help='Image size for training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--eval-interval', type=int, default=1, help='Interval for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu or cuda)')
    return parser.parse_args()

class YOLOloss(nn.Module):
    def __init__(self, num_classes, img_size=416, anchors=None):
        super(YOLOloss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        if anchors is None:
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # Anchors cho feature map nhỏ
                [(30, 61), (62, 45), (59, 119)],     # Anchors cho feature map vừa
                [(116, 90), (156, 198), (373, 326)]  # Anchors cho feature map lớn
            ]
        else:
            self.anchors = anchors
        
        # Hệ số weight cho các thành phần loss
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, predictions, targets, device):
        """
        Args:
            predictions: Danh sách dự đoán 3 scales [p3, p4, p5]
            targets: List of targets [boxes, labels] cho mỗi ảnh trong batch
            device: Thiết bị (cpu hoặc cuda)
        """

        total_loss = 0.0
        coord_loss = 0.0
        conf_loss = 0.0
        cls_loss = 0.0
        num_samples = predictions[0].shape[0]  # Số lượng ảnh trong batch

        # Xử lý từng scale
        for i, (pred, anchors) in enumerate(zip(predictions, self.anchors)):
            batch_size, _, grid_size, _ = pred.size()
            stride = self.img_size // grid_size

            # Reshape dự đoán về [batch, num_anchors, grid_size, grid_size, 5 + num_classes]
            num_anchors = len(anchors)
            pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

            # Get output
            x = torch.sigmoid(pred[..., 0])  # tx
            y = torch.sigmoid(pred[..., 1])  # ty
            w = pred[..., 2]  # tw
            h = pred[..., 3]  # th
            conf = torch.sigmoid(pred[..., 4])  # objectness score
            cls_pred = torch.sigmoid(pred[..., 5:])  # class scores

            # Tạo target tensor
            obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, device=device)
            tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            th = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, self.num_classes, device=device)

            # Convert anhors sang tensor
            anchors_tensor = torch.FloatTensor(anchors).to(device)

            # Xử lý targets
            for b, (boxes, labels) in enumerate(zip(targets['boxes'], targets['labels'])):
                if boxes.size(0) == 0:
                    continue

                # Chuyển boxes từ [0-1] sang pixel coordinates
                boxes_scaled = boxes.clone()
                boxes_scaled[:, 0] *= self.img_size
                boxes_scaled[:, 1] *= self.img_size
                boxes_scaled[:, 2] *= self.img_size
                boxes_scaled[:, 3] *= self.img_size

                # Tim anchor tốt nhất cho mỗi box
                for box_idx, box in enumerate(boxes_scaled):
                    # Lấy tọa độ và kích thước của box
                    gx, gy, gw, gh = box

                    # xác định grid cell
                    gi = int(gx / stride)
                    gj = int(gy / stride)

                    if gi >= grid_size or gj >= grid_size:
                        continue

                    # tính iou với tất cả anchors
                    box_wh = torch.FloatTensor([gw, gh]).to(device)
                    anchor_wh = anchors_tensor * stride

                    # Tính intersection over anchors
                    intersection = torch.min(box_wh, anchor_wh).prod(1)
                    union = box_wh.prod() + anchor_wh.prod(1) - intersection
                    iou = intersection / union

                    # Lấy anchor có iou lớn nhất
                    best_anchor = torch.argmax(iou)

                    # Đánh dấu là Object
                    obj_mask[b, best_anchor, gj, gi] = 1.0
                    noobj_mask[b, best_anchor, gj, gi] = 0.0

                    # Tính target values
                    tx[b, best_anchor, gj, gi] = gx / stride - gi
                    ty[b, best_anchor, gj, gi] = gy / stride - gj
                    tw[b, best_anchor, gj, gi] = torch.log(gw / (anchors_tensor[best_anchor, 0] * stride) + 1e-16)
                    th[b, best_anchor, gj, gi] = torch.log(gh / (anchors_tensor[best_anchor, 1] * stride) + 1e-16)

                    # One-hot encoding class labels
                    tcls[b, best_anchor, gj, gi, labels[box_idx]] = 1.0
            # Tính losses
            # Coodinate loss
            coord_loss += self.lambda_coord * self.mse_loss(x[obj_mask == 1], tx[obj_mask == 1]) / num_samples
            coord_loss += self.lambda_coord * self.mse_loss(y[obj_mask == 1], ty[obj_mask == 1]) / num_samples
            coord_loss += self.lambda_coord * self.mse_loss(w[obj_mask == 1], tw[obj_mask == 1]) / num_samples 
            coord_loss += self.lambda_coord * self.mse_loss(h[obj_mask == 1], th[obj_mask == 1]) / num_samples
            
            # Objectness loss
            conf_loss += self.bce_loss(conf[obj_mask == 1], obj_mask[obj_mask == 1]) / num_samples
            conf_loss += self.lambda_noobj * self.bce_loss(conf[noobj_mask == 1], obj_mask[noobj_mask == 1]) / num_samples

            # Class loss
            cls_loss += self.bce_loss(cls_pred[obj_mask == 1], tcls[obj_mask == 1]) / num_samples
    
        # Tổng hợp loss
        total_loss = coord_loss + conf_loss + cls_loss

        return total_loss, coord_loss, conf_loss, cls_loss\
    
def train(train_loader, model, criterion, optimizer, epoch, device, writer):
    """
    Huấn luyện mô hình cho một epoch
    Args:
        train_loader: DataLoader cho tập huấn luyện
        model: Mô hình FractalNetDetection
        criterion: Hàm loss
        optimizer: Optimizer
        epoch: Số epoch hiện tại
        device: Thiết bị (cpu hoặc cuda)
        write: TensorBoard writer
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    coord_losses = AverageMeter()
    conf_losses = AverageMeter()
    cls_losses = AverageMeter()

    # Chuyển mô hình vào chế độ huấn luyện
    model.train()

    end = time.time()
    pbar = tqdm(train_loader)

    for i, batch in enumerate(pbar):
        data_time.update(time.time() - end)

        # Lấy đầu vào và nhãn
        images = batch['images'].to(device)
        targets = {
            'boxes': batch['boxes'],
            'labels': batch['labels']
        }

        # Forward pass
        outputs = model(images)

        # Tính loss
        loss, (coord_loss, conf_loss, cls_loss) = criterion(outputs, targets, device)
    
        # update meters
        losses.update(loss.item(), images.size(0))
        coord_losses.update(coord_loss.item(), images.size(0))
        conf_losses.update(conf_loss.item(), images.size(0))
        cls_losses.update(cls_loss.item(), images.size(0))

        # Backward pass và cập nhật weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # đo thời gian
        batch_time.update(time.time() - end)
        end = time.time()

        # Cập nhật progress bar
        pbar.set_description(f'Epoch {epoch} - Loss: {losses.avg:.4f}')

    # Ghi tensorboard
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/coord_loss', coord_losses.avg, epoch)
    writer.add_scalar('train/conf_loss', conf_losses.avg, epoch)
    writer.add_scalar('train/cls_loss', cls_losses.avg, epoch)

    return losses.avg

def validate(val_loader, model, criterion, epoch, device, writer, class_name):
    """
    Đánh giá mô hình trên tập validation
    Args:
        val_loader: DataLoader cho tập validation
        model: Mô hình FractalNetDetection
        criterion: Hàm loss
        epoch: Số epoch hiện tại
        device: Thiết bị (cpu hoặc cuda)
        writer: TensorBoard writer
        class_name: Danh sách tên lớp
    """

    batch_time = AverageMeter()
    losses = AverageMeter()

    # Danh sách để lưu các dự đoán và nhãn thực tế
    all_predictions = []
    all_targets = []

    # Chuyển mô hình vào chế độ đánh giá
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader)
        for i, batch in enumerate(pbar):
            # Lấy đầu vào và nhãn
            images = batch['images'].to(device)
            targets = {
                'boxes': batch['boxes'],
                'labels': batch['labels']
            }

            # Forward pass
            outputs = model(images)

            # Tính loss
            loss, _ = criterion(outputs, targets, device)

            # update meters
            losses.update(loss.item(), images.size(0))

            # Lấy dự đoán
            detections = model.detect(images)

            # Tổng hợp dự đoán và nhãn để tính metrics
            for b in range(images.size(0)):
                img_id = batch['img_ids'][b]

                # Xử lý dự đoán
                if detections[b] is not None:
                    for detection in detections[b]:
                        box = detection[:4].cpu()
                        score = detection[4].cpu().item()
                        class_id = torch.argmax(detection[5:]).cpu().item()

                        all_predictions.append([img_id, class_id, score, box[0], box[1], box[2], box[3]])
                
                # Xử lý nhãn thực tế
                boxes = targets['boxes'][b]
                labels = targets['labels'][b]

                if boxes.size(0) > 0:
                    boxes_xyxy = xywh2xyxy(boxes)

                    for box_idx in range(boxes.size(0)):
                        box = boxes_xyxy[box_idx].cpu()
                        label = labels[box_idx].cpu().item()

                        all_targets.append([img_id, label, box[0], box[1], box[2], box[3]])
            
            # đo thời gian
            batch_time.update(time.time() - end)
            end = time.time()

            # Cập nhật progress bar
            pbar.set_description(f'Validation - Loss: {losses.avg:.4f}')

            # Visualize một số dự đoán
            if i % 10 == 0:
                fig = plot_predictions(
                    images[0].cpu(), 
                    detections[0][:, :4] if detections[0] is not None else torch.empty(0, 4),
                    torch.argmax(detections[0][:, 5:], dim=1) if detections[0] is not None else torch.empty(0),
                    class_name)
                
                writer.add_figure(f'val/predictions_epoch_{i}', fig, epoch)
    
    # Tinh toán các chỉ số đánh giá
    precision, recall, mAP, confution_matrix = calculate_metrics(all_predictions, all_targets, iou_threshold=0.5, num_class=len(class_name))

    # Vẽ và lưu confusion matrix
    cm_fig = plot_confusion_matrix(confution_matrix, class_name)
    writer.add_figure('val/confusion_matrix_epoch_{epoch}', cm_fig, epoch)

    # Log to tensorboard
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/precision', precision, epoch)
    writer.add_scalar('val/recall', recall, epoch)
    writer.add_scalar('val/mAP', mAP, epoch)
    
    print(f'Validation Results - Loss: {losses.avg:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {mAP:.4f}')
    
    return losses.avg, mAP

def main():
    args = parse_args()

    # Tạo đường dẫn lưu kết quả
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Tạo dataset và dataloader
    train_dataset = VOCDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        img_size=args.img_size,
        is_train=True
    )

    val_dataset = VOCDataset(
        root_dir=os.path.join(args.data_dir, 'val'),
        img_size=args.img_size,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )

    print(f'Dataset loaded. Train {len(train_dataset)}, Val: {len(val_dataset)}')
    print(f'Classes: {train_dataset.classes}')

    # Tạo mô hình
    model = FractalNetDetection(
        num_classes=len(train_dataset.classes),
        image_size=args.img_size
    ).to(device)

    # Tạo criterion
    criterion = YOLOloss(
        num_classes=len(train_dataset.classes),
        img_size=args.img_size,
    )

    # Tạo optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Tạo learning rate scheduler
    scheduler = optim.lre_scheduler.ReducelLROnPlateau(
        optimizer,
        model='min',
        factor=0.1,
        patience=5,
        verbose=True,
    )

    # Tạo TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Tạo early stopping
    early_stopping = EarlyStopping(patience=10)

    # Tiếp tục huấn luyện từ checkpoint nếu có
    start_epoch = 0
    best_mAP = 0

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f'Loading checkpoint {args.checkpoint}')
            checkpoint = torch.load(args.checkpoint)
            start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded checkpoint (epoch {start_epoch})')
        else:
            print(f'No checkpoint found at {args.checkpoint}')
    
    # Huấn luyện mô hình
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch}/{args.epochs}')
        
        # Huấn luyện
        train_loss = train(train_loader, model, criterion, optimizer, epoch, device, writer)

        # Đánh giá
        if epoch % args.eval_interval == 0:
            print('Validating...')
            val_loss, mAP = validate(val_loader, model, criterion, epoch, device, writer, train_dataset.classes)

        # Cập nhật learning rate
        scheduler.step(val_loss)

        # Lưu checkpoint nếu mAP tốt hơn
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mAP': best_mAP,
            'optimizer': optimizer.state_dict()
        }, is_best, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth.tar'))

        # check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Đóng TensorBoard writer
    writer.close()

    print(f'Best mAP: {best_mAP:.4f}')

if __name__ == '__main__':
    main()    
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import VOCDataset
from model import FractalNetDetection
from utils import (calculate_metrics, plot_predictions, xywh2xyxy, plot_confusion_matrix)

def parse_args():
    parser = argparse.ArgumentParser(description='FractalNet Object Detection Evaluation')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory') # Đường dẫn đến thư mục chứa dữ liệu
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation') # Kích thước batch cho quá trình đánh giá
    parser.add_argument('--img-size', type=int, default=416, help='Image size for evaluation') # Kích thước ảnh đầu vào
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint') # Đường dẫn đến checkpoint của mô hình
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save evaluation results') # Thư mục để lưu kết quả đánh giá
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for DataLoader') # Số lượng worker cho DataLoader
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold for predictions') # Ngưỡng confidence cho các dự đoán
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS threshold for predictions') # Ngưỡng NMS cho các dự đoán
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IoU threshold for mAP calculation') # Ngưỡng IoU cho việc tính toán mAP
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (cpu or cuda)') # Thiết bị để sử dụng cho quá trình đánh giá
    return parser.parse_args()

def evaluate(model, dataloader, device, conf_thres, nms_thres, iou_thres, class_name, output_dir=None):
    """
    Hàm đánh giá mô hình trên tập dữ liệu
    Args:
        model: mô hình FractalNetDetection đã được huấn luyện
        dataloader: DataLoader cho tập dữ liệu kiểm tra
        device: thiết bị để thực hiện đánh giá (cpu hoặc cuda)
        conf_thres: ngưỡng confidence cho các dự đoán
        nms_thres: ngưỡng NMS cho các dự đoán
        iou_thres: ngưỡng IoU cho việc tính toán mAP
        class_name: tên các lớp trong tập dữ liệu
        output_dir: thư mục để lưu kết quả đánh giá (nếu có)
    Returns:
        metrics: dictionary chứa các chỉ số đánh giá (precision, recall, mAP, v.v.)
    """
    # Đưa mô hình vào chế độ đánh giá
    model.eval()

    # Khởi tạo danh sách để lưu các dự đoán và nhãn thực tế
    all_predictions = []
    all_targets = []

    # Duyệt qua từng batch trong DataLoader
    for images, targets in tqdm(dataloader, desc='Evaluating', unit='batch'):
        images = images.to(device) # Chuyển ảnh vào thiết bị (cpu hoặc cuda)
        targets = targets.to(device) # Chuyển nhãn vào thiết bị

        # Dự đoán từ mô hình
        with torch.no_grad():
            pbar = tqdm(dataloader)
            for i, batch in enumerate(pbar):

                images = batch['images'].to(device)
                targets = {
                    'boxes': batch['boxes'],
                    'labels': batch['labels']
                }

                # Lấy dự đoán
                detections = model.detect(images, conf_thres=conf_thres, nms_thres=nms_thres)

                # Collect dự đoán và nhãn thực tế để tính toán các chỉ số
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
                        # Chuyển đổi từ xywh sang xyxy
                        boxes_xyxy = xywh2xyxy(boxes)

                        for box_idx in range(boxes.size(0)):
                            box = boxes_xyxy[box_idx].cpu()
                            class_id = labels[box_idx].cpu().item()

                            all_targets.append([img_id, class_id, box[0], box[1], box[2], box[3]])
                
                # Lưu visualization nếu output_dir được chỉ định
                if output_dir is not None:
                    for b in range(images.size(0)):
                        img_id = batch['img_ids'][b]

                        # tạo figure
                        fig = plot_predictions(
                            images[b].cpu(),
                            detections[b][:, :4] if detections[b] is not None else torch.empty(0, 4),
                            torch.argmax(detections[b][:, 5:], dim=1) if detections[b] is not None else torch.empty(0),
                            class_name
                        )

                        # Lưu figure
                        fig.savefig(os.path.join(output_dir, f'{img_id}.png'))
                        plt.close(fig)
    
    # Tính toán các chỉ số đánh giá
    precision, recall, mAP, confusion_matrix = calculate_metrics(
        all_predictions, all_targets, iou_threshold=iou_thres, num_class=len(class_name)
    )

    # Vẽ confusion matrix
    if output_dir is not None:
        plot_confusion_matrix(confusion_matrix, class_name,
                              os.path.join(output_dir, 'confusion_matrix.png'))
    
    return precision, recall, mAP, confusion_matrix

def main():
    args = parse_args()

    # Kiểm tra và tạo thư mục đầu ra nếu chưa tồn tại
    if args.output_dỉ:
        os.makedirs(args.output_dir, exist_ok=True)

    # thiet lập thiết bị
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #Tạo DataLoader cho tập dữ liệu kiểm tra
    test_dataset = VOCDataset(
        data_dir=args.data_dir,
        img_size=args.img_size,
        is_train=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True
    )

    print(f'Test dataset loaded. Size: {len(test_dataset)}')
    print(f'Classes: {test_dataset.classes}')

    # Loat model
    checkpoint = torch.load(args.checkpoint)
    model = FractalNetDetection(
        num_classes=len(test_dataset.classes),
        img_size=args.img_size
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'Model loaded from {args.checkpoint}')

    # Đánh giá mô hình
    precision, recall, mAP, confusion_matrix = evaluate(
        model,
        test_dataloader,
        device,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        iou_thres=args.iou_thres,
        class_name=test_dataset.classes,
        output_dir=args.output_dir
    )
    print(f'Evaluation Results:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  mAP@{args.iou_thres}: {mAP:.4f}')

    # Ghi kết quả vào file
    if args.output_dir:
        with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'mAP@{args.iou_thres}: {mAP:.4f}\n')

if __name__ == '__main__':
    main()
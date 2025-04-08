import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import time
import seaborn as sns
import pandas as pd
import shutil

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Tính toán IoU giữa hai bounding box.
    Args:
        box1: (tensor) box thứ nhất
        box2: (tensor) box thứ hai
        x1y1x2y2: (bool) Định dạng của nhãn là [x1, y1, x2, y2] hay [x, y, w, h]
    Returns:
        iou: (tensor) Giá trị IoU
    """

    if not x1y1x2y2:
        # Chuyển đổi từ [x, y, w, h] sang [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Tính diện tích giao và hợp
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    union_area = b1_area + b2_area - inter_area + 1e-16

    iou = inter_area / union_area

    return iou

def xywh2xyxy(x):
    """
    Chuyển từ [x, y, w, h] sang [x1, y1, x2, y2]
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh(x):
    """
    Chuyển từ [x1, y1, x2, y2] sang [x, y, w, h]
    """
    y = x.new(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def compute_ap(recall, precision):
    """
    Tính toán AP từ độ chính xác và độ nhạy
    Args:
        recall: (tensor) Độ nhạy
        precision: (tensor) Độ chính xác
    Returns:
        ap: (float) Giá trị Average Precision
    """
    # Thêm điểm đầu và cuối cho recall và precision
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Tính toán độ chính xác trung bình
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Tìm chỉ số của các điểm recall khác nhau
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
     # Tính AP = ∑ (R_n - R_{n-1}) * P_n
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_metrics(pred_boxes, true_boxes, iou_threshold=0.5, num_class=30):
    """
    Tính precision, recall và mAP
    Args:
        pred_boxes: (list) Danh sách predicted boxes [img_idx, class_pred, conf, x1, y1, x2, y2]
        true_boxes: (list) Danh sách ground truth boxes [img_idx, class_gt, x1, y1, x2, y2]
        iou_threshold: (float) Ngưỡng IoU để xác định positive prediction
        num_class: (int) Số lượng lớp
    Returns:
        precision: (float) Giá trị precision
        recall: (float) Giá trị recall
        mAP: (float) Giá trị mean Average Precision
        confusion_matrix: (numpy array) Ma trận nhầm lẫn
    """

    # Khởi tạo các biến
    true_positives = []
    scores = []
    pred_classes = []

    # Số lượng ground truth boxes cho mỗi lớp
    n_gt = defaultdict(int)
    for box in true_boxes:
        n_gt[int(box[1])] += 1
    
    # Sắp xếp prediction theo confidence giảm dần
    pred_boxes = sorted(pred_boxes, key=lambda x: x[2], reverse=True)

    # Đánh dấu các GT boxes dã được match
    detected_boxes = defaultdict(list)

    # Khởi tạo confusion matrix
    confution_matrix = np.zeros((num_class, num_class), dtype=np.int64)

    # Duyệt qua từng predicted box
    for pred_box in pred_boxes:
        img_idx, class_pred, conf, x1, y1, x2, y2 = pred_box
        class_pred = int(class_pred)

        # lưu lại score vaf class prediction
        scores.append(conf)
        pred_classes.append(class_pred)

        # Tìm tất cả GT boxes cho ảnh
        img_gt_boxes = [box for box in true_boxes if box[0] == img_idx]

        best_iou = 0
        best_gt_idx = -1
        best_gt_class = -1

        # Nếu không có GT box trong ảnh
        if len(img_gt_boxes) == 0:
            true_positives.append(0)
            continue

        # Tìm GT box có iou lớn nhất với predicted box
        for gt_idx, gt_box in enumerate(img_gt_boxes):
            _, gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            gt_class = int(gt_class)

            #Tính iou
            iou = bbox_iou(
                torch.tensor([[x1, y1, x2, y2]]),
                torch.tensor([[gt_x1, gt_y1, gt_x2, gt_y2]])
            )[0]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                best_gt_class = gt_class
        
        # Cập nhật confusion matrix
        if best_gt_class != -1:
            confution_matrix[best_gt_class][class_pred] += 1
        
        # Kiểm tra nếu iou vượt ngưỡng và GT box chưa được match
        if best_iou >= iou_threshold and best_gt_class == class_pred and best_gt_idx not in detected_boxes[(img_idx, class_pred)]:
            true_positives.append(1)
            detected_boxes[(img_idx, class_pred)].append(best_gt_idx)
        else:
            true_positives.append(0)
        
    # Tính toán precision và recall
    true_positives = np.array(true_positives)
    false_positives = 1 - true_positives

    # Tính tổng lũy tiến (cumulative sums)
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    # Tính precision và recall cho từng ngưỡng
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
    recalls = tp_cumsum / (sum(n_gt.values()) + 1e-16)

    # Tính average precision (AP)
    ap = compute_ap(recalls, precisions)

    # Tính precision và recall cuối cùng
    precisions = precisions[-1] if len(precisions) > 0 else 0
    recall = recalls[-1] if len(recalls) > 0 else 0

    return precisions, recall, ap, confution_matrix

def plot_confusion_matrix(confusion_matrix, class_names, figsize=(10, 8), output_path=None):
    """
    Vẽ confusion matrix
    Args:
        confusion_matrix: Confusion matrix
        class_names: Danh sách tên các classes
        figsize: Kích thước figure
        output_path: Đường dẫn để lưu figure
    """
    # Tạo dataframe từ confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

    # Vẽ heatmap
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    # thêm tiêu đề và nhãn
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)

    # Lưu figure nếu output_path được cung cấp
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
    return heatmap.get_figure()

def plot_predictions(image, boxes, labels, class_names, figsize=(10, 10)):
    """
    Vẽ các bounding box trên ảnh
    Args:
        image: Tensor ảnh [C, H, W]
        boxes: Tensor bounding box [N, 4], định dạng [x1, y1, x2, y2]
        labels: Tensor labels [N]
        class_names: Danh sách tên các lớp
        figsize: Kích thước figure
    """
    
    # Chuyển đổi từ tensor sang numpy
    image = image.permute(1, 2, 0).cpu().numpy()

    # Giải chuẩn ảnh (Denormalize image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Vẽ ảnh
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # Vẽ các bounding box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.cpu().numpy()
        w, h = x2 - x1, y2 - y1

        # Tạo rectangle
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Thêm nhãn
        class_name = class_names[labels[i]]
        ax.text(x1, y1, class_name, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    return fig

class AverageMeter(object):
    """
    Tính toán giá trị trung bình của một biến trong quá trình huấn luyện
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Lưu checkpoint
    Args:
        state: Trạng thái của mô hình
        is_best: Kiểm tra xem đây có phải là mô hình tốt nhất không
        filename: Tên file để lưu checkpoint
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    print(f"Checkpoint saved to {filename}")

class EarlyStopping:
    """
    Dừng training sớm khi validation loss không giảm
    """

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fractalnet import FractalNet, FractalBlock, Flatten
from utils import bbox_iou

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Mỗi anchor box dự đoán: tx, ty, tw, th, objetctness, class scores
        self.out_channels = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        # x: [batch_size, in_channels, height, width]
        x = self.conv(x)

class FractalNetDetection(nn.Module):
    def __init__(self, num_classes, img_size=416, anchors=None):
        super(FractalNetDetection, self).__init__()
        
        if anchors is None:
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # Anchors cho feature map nhỏ
                [(30, 61), (62, 45), (59, 119)],     # Anchors cho feature map vừa
                [(116, 90), (156, 198), (373, 326)]  # Anchors cho feature map lớn
            ]
        else:
            self.anchors = anchors
        
        self.num_classes = num_classes
        self.img_size = img_size

        # Backbone FractalNet
        data_shape = (3, img_size, img_size, num_classes)
        n_columns = 3
        init_channels = 64
        p_ldrop = 0.3
        dropout_probs = [0.0, 0.1, 0.2, 0.3, 0.4]  # 5 blocks
        gdrop_ratio = 0.5

        self.backbone = FractalNet(
            data_shape=data_shape,
            n_columns=n_columns,
            init_channels=init_channels,
            p_ldrop=p_ldrop,
            dropout_probs=dropout_probs,
            gdrop_ratio=gdrop_ratio,
            gap=0,
            doubling=True
        )

        # Lấy feature maps từ 3 scales khác nhau
        self.out_channels = [init_channels * (2 ** i) for i in range(len(dropout_probs))]

        # Detection heads cho từng scale
        self.detection_head_1 = DetectionHead(self.out_channels[2], num_classes, len(self.anchors[0]))
        self.detection_head_2 = DetectionHead(self.out_channels[3], num_classes, len(self.anchors[1]))
        self.detection_head_3 = DetectionHead(self.out_channels[4], num_classes, len(self.anchors[2]))

        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Leteral connections
        self.lateral_3_4 = nn.Conv2d(self.out_channels[4], self.out_channels[3], kernel_size=1)
        self.lateral_2_3 = nn.Conv2d(self.out_channels[3], self.out_channels[2], kernel_size=1)

    def forward(self, x):
        features = []
        global_cols = None

        # Lấy các feature maps từ backbone
        for i, layer in enumerate(self.backbone.layers):
            if isinstance(layer, nn.MaxPool2d):
                features.append(x) # Lưu lại feature map trước khi pooling
            
            if isinstance(layer, FractalBlock):
                if not self.backbone.consist_gdrop or global_cols is None:
                    GB = int(x.size(0) * self.backbone.gdrop_ratio)
                    global_cols = np.random.randint(0, self.backbone.n_columns, size=[GB])
                x = layer(x, global_cols, deepest=False)
            
            else: 
                if not isinstance(layer, nn.Linear) and not isinstance(layer, Flatten):
                    x = layer(x)
        
        # Lấy 3 feature maps cho 3 scales
        c3 = features[2]  # Small scale
        c4 = features[3]  # Medium scale
        c5 = features[4]  # Large scale

        # FPN-like architecture
        p5 = self.detection_head_3(c5)
        
        p5_up = self.upsample(self.lateral_3_4(c5))
        p4 = self.detection_head_2(c4 + p5_up)
        
        p4_up = self.upsample(self.lateral_2_3(c4 + p5_up))
        p3 = self.detection_head_1(c3 + p4_up)
        
        return [p3, p4, p5]
    
    def _make_grid(self, nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
    
    def _decode_pred(self, pred, anchors, stride):
        """
        Giải mã dự đoán đẻ lấy bounding box
        """

        batch_size = pred.size(0)
        num_anchors = len(anchors)
        grid_size = pred.size(2)

        # Reshape dự đoán về dạng [batch_size, num_anchors, grid_size, grid_size, 5 + num_classes]
        pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        # Lấy outputs
        x = torch.sigmoid(pred[..., 0]) # center x
        y = torch.sigmoid(pred[..., 1]) # center y
        w = pred[..., 2] # width
        h = pred[..., 3] # height
        conf = torch.sigmoid(pred[..., 4]) # objectness score
        cls_pred = torch.sigmoid(pred[..., 5:]) # Dự đoán lớp

        # Tính offsets cho mỗi grid
        grid = self._make_grid(grid_size, grid_size).to(pred.device)

        # Thêm offset vào center x, y
        x = x + grid[..., 0]
        y = y + grid[..., 1]

        # Scale w, h theo anchors
        anchors = torch.FloatTensor(anchors).to(pred.device)
        anchors = anchors.view(1, num_anchors, 1, 1, 2)
        w = torch.exp(w) * anchors[..., 0]
        h = torch.exp(h) * anchors[..., 1]

        # Scale x, y về kích thước ảnh gốc
        x = x * stride / self.img_size
        y = y * stride / self.img_size
        w = w * stride / self.img_size
        h = h * stride / self.img_size

        # Trả về kết quả x, y, w, h, conf, cls_pred
        return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1), \
                        conf.unsqueeze(-1), cls_pred), dim=-1)
    
    def detect(self, x, conf_thres=0.5, nms_thres=0.4):
        """
        Phát hiện các đối tượng trong hình ảnh đầu vào
        Args:
            x: tensor ảnh đầu vào [batch_size, 3, H, W]
            conf_thres: Confidence threshold để lọc các dự đoán
            nms_thres: non-maximum suppression threshold
        Returns:
            detections: list các dự đoán của mỗi ảnh trong batch
        """
        # Đưa x vào model để lấy dự đoán
        preds = self.forward(x)

        # Xử lý mỗi scale
        output = []
        strides = [self.img_size // pred.size(2) for pred in preds] # [8, 16, 32] for img_size=416

        for i, (pred, anchors, stride) in enumerate(zip(preds, self.anchors, strides)):
            # Decode predictions
            pred = self._decode_pred(pred, anchors, stride)
            output.append(pred.view(x.size(0), -1, 5 + self.num_classes))

        # Concatenate dự đoán từ tất các scales
        output = torch.cat(output, dim=1)

        # Sử dụng NMS
        batch_detections = []
        
        for i in range(output.size(0)):
            detections = output[i] # Lấy dự đoán cho từng ảnh trong batch
            detections = detections[detections[..., 4] > conf_thres] # Lọc dự đoán theo confidence score

            if not detections.size(0):
                batch_detections.append(None)
                continue

            # Lấy các bounding box, scores và class scores
            scores = detections[..., 4] * detections[..., 5:].max(1)[0]
            
            # Sắp xếp các dự đoán theo confidence
            _, sort_idx = scores.sort(descending=True)
            detections = detections[sort_idx]

            # Convert định dạng sang xyxy cho NMS
            box_corner = detections.new(detections.shape)
            box_corner[..., 0] = detections[..., 0] - detections[..., 2] / 2
            box_corner[..., 1] = detections[..., 1] - detections[..., 3] / 2
            box_corner[..., 2] = detections[..., 0] + detections[..., 2] / 2
            box_corner[..., 3] = detections[..., 1] + detections[..., 3] / 2   
            detections[..., :4] = box_corner[..., :4]

            # Thực hiện NMS
            keep = []
            while detections.size(0):
                large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
                label_match = detections[0, -1] == detections[:, -1]
                invalid = large_overlap & label_match
                weights = detections[invalid, 4:5]
                detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
                keep.append(detections[0])
                detections = detections[~invalid]

            if keep:
                batch_detections.append(torch.stack(keep))
            else:
                batch_detections.append(None)
        
        return batch_detections
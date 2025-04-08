import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_size=1248, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Thư mục chứa ảnh và annotations XML
            img_size (int): Kích thước ảnh đầu vào
            transform (callable, optional): Transform tùy chọn
            is_train (bool): Chế độ train hoặc val
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        # Lấy danh sách các ảnh và annotations
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations')
        self.image_files = [f for f in os.listdir(self.image_dir) 
                            if f.endswith('.jpg', '.png', '.jpeg')]
        # Chỉ giữ lại các ảnh có file XML tương ứng
        self.image_files = [f for f in self.image_files 
                            if os.path.exists(os.path.join(self.annot_dir, f.replace(os.path.splitext(f)[1], '.xml')))]
        
        # Lấy danh sách các classes từ tất cả các file XML
        self.classes = self._get_classes()

        # Mapping từ tên class sang index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Chuẩn bị transforms
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Augmentations cho training
        self.train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def _get_classes(self):
        """
        Lấy danh sách các classes từ tất cả các file XML trong thư mục annotations
        """
        classes = []
        for img_file in self.image_files:
            xml_file = os.path.join(self.annot_dir, 
                                  os.path.splitext(img_file)[0] + '.xml')
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    classes.append(cls_name)
        
        classes.sort()
        return classes
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        xml_path = os.path.join(self.annot_dir, 
                              os.path.splitext(img_file)[0] + '.xml')
        
        # Đọc ảnh
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Áp dụng augmentations nếu là training
        if self.is_train:
            image = self.train_transforms(image)
        
        # Áp dụng transform
        image = self.transform(image)
        
        # Đọc annotations từ file XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in self.class_to_idx:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Chuẩn hóa tọa độ về [0, 1]
            xmin /= orig_width
            xmax /= orig_width
            ymin /= orig_height
            ymax /= orig_height
            
            # Chuyển sang format YOLO (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            
            # Thêm vào danh sách
            boxes.append([x_center, y_center, width, height])
            labels.append(self.class_to_idx[cls_name])
        
        # Chuyển sang tensor
        if len(boxes) > 0:
            boxes = torch.FloatTensor(boxes)
            labels = torch.LongTensor(labels)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': os.path.splitext(img_file)[0]
        }
    
    def collate_fn(self, batch):
        """
        Custom collate function để xử lý batch có số lượng objects khác nhau
        """
        images = []
        boxes = []
        labels = []
        image_ids = []
        
        for b in batch:
            images.append(b['image'])
            boxes.append(b['boxes'])
            labels.append(b['labels'])
            image_ids.append(b['image_id'])
            
        images = torch.stack(images, dim=0)
        
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'image_ids': image_ids
        }
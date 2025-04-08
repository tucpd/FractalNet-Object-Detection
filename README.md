# FractalNet Object Detection

Model use backbone is FractalNet and provide code to training model for object detection task. Model use the metrics precision, recall and mAP to evaluate and confusion matrix to detail analysis results. 

## Structure
project/ 
├── fractalnet.py # FractalNet 
├── model.py # FractalNet model for object detection 
├── dataset.py # Xử lý dữ liệu định dạng Pascal VOC 
├── utils.py # Các hàm tiện ích 
├── train.py # Script huấn luyện 
├── evaluate.py # Script đánh giá 
├── config.py # System config
├── voc_to_xml.py # Tool convert format 
└── README.md 

## Dataset prepare
The model use data pascal VOC format with folder structure:
data/ 
├── train/ 
    │ ├── images/ 
    │ │ ├── image1.jpg 
    │ │ ├── image2.jpg 
    │ │ └── ... 
    │ └── annotations/ 
        │ ├── image1.xml 
        │ ├── image2.xml 
        │ └── ... 
├── val/ 
    │ ├── images/ 
    │ │ └── ... 
    │ └── annotations/ 
    │ └── ... 
└── test/ 
    ├── images/ 
        │ └── ... 
    └── annotations/ 
        └── ...

Each file xml contain info about object in image, include class name and bounding box.

### Convert dataset structure
Using tool `voc_to_xml.py` to convert structure dataset
```bash
python voc_to_xml.py --input-dir /path/to/your/data --output-dir data --dataset train
python voc_to_xml.py --input-dir /path/to/your/val_data --output-dir data --dataset val
python voc_to_xml.py --input-dir /path/to/your/test_data --output-dir data --dataset test


# FractalNet Object Detection

Model use backbone is FractalNet and provide code to training model for object detection task. Model use the metrics precision, recall and mAP to evaluate and confusion matrix to detail analysis results. 

## Structure
```
project/
├── fractalnet.py        # FractalNet backbone implementation
├── model.py             # FractalNet object detection model
├── dataset.py           # Pascal VOC dataset handling
├── utils.py             # Utility functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── config.py            # System configuration
├── voc_to_xml.py        # Dataset format conversion tool
└── README.md            # Project documentation
```

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
```
## Training model
Script training model:
```bash
python train.py --data-dir data --batch-size 8 --epochs 100 --lr 0.001 --img-size 416 --log-dir logs --save-dir checkpoints
```

### The parameter:
```bash
--data-dir: Directory containing the data
--batch-size: Batch size
--epochs: Number of epochs
--lr: Learning rate
--img-size: Input image size
--log-dir: Directory for Tensorboard logs
--save-dir: Directory for saving checkpoints
--checkpoint: Path to checkpoint for resuming training (optional)
```

## Evaluate model
Script evaluate model:
```bash
python evaluate.py --data-dir data/test --checkpoint checkpoints/model_best.pth.tar --output-dir output --conf-thres 0.5 --nms-thres 0.4 --iou-thres 0.5
```

The parameter:
```bash
--data-dir: Directory containing test data
--checkpoint: Path to model checkpoint
--output-dir: Directory for saving evaluation results
--conf-thres: Confidence threshold
--nms-thres: Non-maximum suppression threshold
--iou-thres: IoU threshold for mAP calculation
```

## Track the training process with Tensorboard
```bash
tensorboard --logdir logs
```
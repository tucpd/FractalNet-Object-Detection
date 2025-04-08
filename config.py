import os
import importlib.util
import sys

def get(config_path):
    """ 
    Đọc file config và trả về module config
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Cấu hình mặc định
# Mô hình
MODEL = {
    'NUM_CLASSES': 20,  # Số lượng lớp trong mô hình
    'IMG_SIZE': 416,  # Kích thước ảnh đầu vào
    'ANCHORS': [
        [(10, 13), (16, 30), (33, 23)],  # Anchors cho feature map nhỏ
        [(30, 61), (62, 45), (59, 119)],  # Anchors cho feature map vừa
        [(116, 90), (156, 198), (373, 326)]  # Anchors cho feature map lớn
    ],
    'FRACTALNET':{
        'N_COLUMNS': 3,
        'INIT_CHANNELS': 64,
        'P_LDROP': 0.3,
        'DROPOUT_PROBS': [0.0, 0.1, 0.2, 0.3, 0.4],  # 5 blocks
        'GDROP_RATIO': 0.5,
        'DOUBLING': True,
        'DROPOUT_POS': 'CDBR'
    }
}

# Huấn luyện
TRAIN = {
    'BATCH_SIZE': 8,
    'EPOCHS': 100,
    'LR' : 0.001,
    'WEIGHT_DECAY': 0.0005,
    'LR_SCHEDULER': {
        'TYPE': 'ReduceLROnPlateau',
        'PATIENCE': 5,
        'FACTOR': 0.1,
    },
    'EARLY_STOPPING': {
        'PATIENCE': 10,
        'MIN_DELTA': 0.0,
    }
}

# dataset
DATA = {
    'TRAIN_DIR': 'data/train',
    'VAL_DIR': 'data/val',
    'TEST_DIR': 'data/test',
    'NUM_WORKERS': 4,
}

# Đánh giá
EVAL = {
    'CONF_THRESHOLD': 0.5,
    'NMS_THRESHOLD': 0.4,
    'IOU_THRESHOLD': 0.5,
    'EVAL_INTERVAL': 1,  # Số lần đánh giá trong mỗi epoch
}

# Logging
LOG = {
    'LOG_DIR': 'logs',
    'SAVE_DIR': 'checkpoints',
    'OUTPUT_DIT': 'output'
}
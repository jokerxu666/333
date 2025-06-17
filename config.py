import os

# 数据集配置
DATASET_PATH = 'data'
IMAGE_SIZE = 128  # 减小图像大小以加快训练
NUM_CLASSES = 3   # KiTS19 数据集有 3 个类别（背景、肾脏、肿瘤）

# 训练配置
BATCH_SIZE = 2    # 减小 batch size
NUM_WORKERS = 2   # 减少工作进程数
NUM_EPOCHS = 3    # 减少训练轮数用于测试
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# 模型配置
IN_CHANNELS = 1   # CT 图像是单通道的
HIDDEN_SIZE = 256  # 减小模型大小
MLP_SIZE = 1024
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

# 路径配置
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

# 创建必要的目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class Config:
    # 数据集配置
    DATASET_PATH = DATASET_PATH
    IMAGE_SIZE = IMAGE_SIZE
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # 数据增强配置
    AUGMENTATION = {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotate': True,
        'rotate_limit': 45,
        'brightness_contrast': True,
        'gaussian_noise': True,
        'gaussian_blur': True
    }
    
    # 模型配置
    MODEL = {
        'in_channels': IN_CHANNELS,
        'out_channels': NUM_CLASSES,
        'img_size': IMAGE_SIZE,
        'patch_size': 16,
        'embed_dim': HIDDEN_SIZE,
        'depth': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'mlp_ratio': MLP_SIZE / HIDDEN_SIZE,
        'qkv_bias': True,
        'drop_rate': DROPOUT,
        'attn_drop_rate': DROPOUT
    }
    
    # 训练配置
    TRAIN = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'warmup_epochs': 5,
        'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        'num_workers': NUM_WORKERS
    }
    
    # 优化器配置
    OPTIMIZER = {
        'type': 'Adam',
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    
    # 学习率调度器配置
    SCHEDULER = {
        'type': 'CosineAnnealingLR',
        'T_max': NUM_EPOCHS,
        'eta_min': 1e-6
    }
    
    # 损失函数配置
    LOSS = {
        'type': 'BCEWithLogitsLoss',
        'pos_weight': 1.0
    }
    
    # 保存配置
    SAVE = {
        'checkpoint_dir': CHECKPOINT_DIR,
        'log_dir': LOG_DIR,
        'save_interval': 5  # 每5个epoch保存一次
    }
    
    # 评估配置
    EVAL = {
        'threshold': 0.5,
        'metrics': ['dice', 'iou']
    } 
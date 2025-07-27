# cv_foundry/blueprints/dino_game/config.py (V2 - 增加dino类别)
# ... (前面的路径定义保持不变) ...

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BLUEPRINT_DIR = Path(__file__).resolve().parent

# --- 1. 资产与类别定义 (V2) ---
ASSETS_PATH = BLUEPRINT_DIR / "assets"
CLASSES = {
    "cactus": 0,
    "dino": 1, # <--- 新增！现在模型需要认识两个东西
}

# --- 2. 数据合成器配置 (V2) ---
SYNTHESIS_CONFIG = {
    "NUM_TRAIN_IMAGES": 1500, # 增加了类别，我们最好也增加数据量
    "NUM_VAL_IMAGES": 300,
    "IMAGE_WIDTH": 600,
    "IMAGE_HEIGHT": 150,
    # V2 新增: 强制每张图必须有且只有一个dino
    "FORCE_DINO": True, 
    "MAX_OBSTACLES_PER_IMAGE": 3,
    "SCALE_RANGE": (0.8, 1.2),
}

# --- 3 & 4. 模型和训练器配置 (保持不变) ---
MODEL_CONFIG = {"BASE_MODEL": "yolo11n.pt"} # 或者您下载的yolov8n.pt
TRAINER_CONFIG = {"EPOCHS": 50, "BATCH_SIZE": 16, "IMG_SIZE": 320}
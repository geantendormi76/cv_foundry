# cv_foundry_lib/blueprints/dino_game/config.py (黄金标准 V2.0)

from pathlib import Path

# --- 1. 核心路径定义 (Path Definition) ---
# 这是整个框架路径感知的基石。
# Path(__file__) 获取当前文件(config.py)的路径
# .resolve() 将其转换为绝对路径
# .parents[3] 向上追溯三级父目录，找到项目根目录 (dino_game -> blueprints -> cv_foundry_lib -> PROJECT_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLUEPRINT_DIR = Path(__file__).resolve().parent

# 基于项目根目录，定义清晰的输入/输出路径
INPUTS_DIR = PROJECT_ROOT / "_inputs"
OUTPUTS_DIR = PROJECT_ROOT / "_outputs"

# --- 2. 蓝图专属资产与类别定义 (Assets & Classes) ---
ASSETS_PATH = BLUEPRINT_DIR / "assets" # 路径已更新，指向蓝图内部的assets
CLASSES = {
    # 顺序必须与Roboflow导出的data.yaml严格一致！
    "bird": 0,
    "cactus": 1,
    "dino": 2,
}

# --- 3. 数据合成器配置 (Data Synthesizer Config) ---
SYNTHESIS_CONFIG = {
    "NUM_TRAIN_IMAGES": 1500,
    "NUM_VAL_IMAGES": 300,
    "IMAGE_WIDTH": 600,
    "IMAGE_HEIGHT": 150,
    "FORCE_DINO": True,
    "MAX_OBSTACLES_PER_IMAGE": 3,
    "SCALE_RANGE": (0.8, 1.2),
    # [新] 输出路径现在由配置驱动
    "OUTPUT_DATASET_DIR": OUTPUTS_DIR / "datasets" / BLUEPRINT_DIR.name
}

# --- 4. 模型与训练器配置 (Model & Trainer Config) ---
MODEL_CONFIG = {
    # [路径更新] 明确指向_inputs目录中的基础模型
    "BASE_MODEL": INPUTS_DIR / "base_models" / "yolov8n.pt"
}

TRAINER_CONFIG = {
    "EPOCHS": 50,
    "BATCH_SIZE": 16,
    "IMG_SIZE": 320,
    "OUTPUT_MODELS_DIR": OUTPUTS_DIR / "models",
    
    # [新] 为微调阶段添加专门的配置
    "FINETUNE_CONFIG": {
        "EPOCHS": 75, # 微调时，我们可以增加轮次以更好地学习真实数据特征
        "BATCH_SIZE": 8 # 使用更小的批量，让模型更精细地学习每一张宝贵的真实图片
    },
    # [新] 定义真实数据集的位置
    "REAL_DATASET_DIR": INPUTS_DIR / "real_world_data" / "annotated_data"
}
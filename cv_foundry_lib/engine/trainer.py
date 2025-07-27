# cv_foundry/foundry_engine/trainer.py

import yaml
from pathlib import Path
from typing import Type

from ultralytics import YOLO

def _create_dataset_yaml(config_module: Type):
    """
    动态创建一个YOLOv8需要的dataset.yaml文件。
    这个文件告诉YOLO在哪里找到训练和验证数据，以及类别信息。
    """
    blueprint_name = config_module.BLUEPRINT_DIR.name
    dataset_root = config_module.BASE_DIR / "datasets" / blueprint_name
    
    # 将Python字典转换为YAML格式的字符串
    yaml_content = {
        'path': str(dataset_root.resolve()),  # 数据集的绝对路径
        'train': 'train/images',              # 训练图片目录 (相对于path)
        'val': 'val/images',                  # 验证图片目录 (相对于path)
        'names': {k: v for v, k in config_module.CLASSES.items()} # 类别ID到名称的映射
    }
    
    # 将YAML内容写入到数据集的根目录
    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"动态生成数据集配置文件: {yaml_path}")
    return yaml_path

def run(config_module: Type):
    """模型训练引擎的主入口。"""
    print("\n--- 启动模型训练引擎 (Trainer) ---")
    
    # 1. 创建YOLO需要的数据集配置文件
    try:
        # PYYAML是ultralytics的依赖，所以我们不需要单独安装
        dataset_yaml_path = _create_dataset_yaml(config_module)
    except Exception as e:
        print(f"[致命错误] 创建dataset.yaml失败: {e}")
        return

    # 2. 初始化YOLO模型
    model_cfg = config_module.MODEL_CONFIG
    try:
        # 根据蓝图配置，加载YOLOv8n作为基础模型
        model = YOLO(model_cfg['BASE_MODEL'])
        print(f"成功初始化基础模型: {model_cfg['BASE_MODEL']}")
    except Exception as e:
        print(f"[致命错误] 初始化YOLO模型失败: {e}")
        return

    # 3. 启动训练
    trainer_cfg = config_module.TRAINER_CONFIG
    blueprint_name = config_module.BLUEPRINT_DIR.name
    models_dir = config_module.BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True) # 确保模型输出目录存在
    
    print("\n--- 开始模型训练 ---")
    print("  这可能需要一些时间，具体取决于您的硬件和训练配置...")
    print("  您可以在终端中看到YOLO的实时训练日志。")
    
    try:
        # 这是核心训练调用
        model.train(
            data=str(dataset_yaml_path),
            epochs=trainer_cfg['EPOCHS'],
            batch=trainer_cfg['BATCH_SIZE'],
            imgsz=trainer_cfg['IMG_SIZE'],
            project=str(models_dir),    # 将训练结果保存到我们的models目录
            name=f"{blueprint_name}_train_results", # 在models下创建一个子目录
            exist_ok=True # 如果目录已存在，则覆盖
        )
        print("\n✅ 模型训练成功完成！")

        # 4. 将训练好的最佳模型复制到根模型目录
        # ultralytics会自动将最好的模型保存为 best.pt
        best_model_path = models_dir / f"{blueprint_name}_train_results/weights/best.pt"
        target_path = models_dir / f"{blueprint_name}.pt"
        best_model_path.rename(target_path)
        print(f"  > 最佳模型已复制到: {target_path}")

    except Exception as e:
        print(f"\n[致命错误] 模型训练过程中发生错误: {e}")
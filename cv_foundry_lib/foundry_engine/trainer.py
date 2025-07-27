# cv_foundry/foundry_engine/trainer.py (黄金标准 V2.0 - 重构版)

import yaml
from pathlib import Path
from typing import Type

from ultralytics import YOLO

# cv_foundry/foundry_engine/trainer.py (黄金标准 V2.1 - 两阶段版)

import yaml
from pathlib import Path
from typing import Type

from ultralytics import YOLO


def _create_dataset_yaml(dataset_root: Path, config_module: Type) -> Path:
    """ 智能地为指定数据集根目录创建dataset.yaml文件。"""
    
    # 智能判断验证集目录名
    if (dataset_root / "valid").is_dir():
        val_dir = "valid/images"
    else:
        val_dir = "val/images" # 默认为合成数据的 'val'

    # 智能判断测试集目录是否存在
    test_dir = 'test/images' if (dataset_root / "test").is_dir() else None

    yaml_content = {
        'path': str(dataset_root.resolve()),
        'train': 'train/images',
        'val': val_dir,
        # 从配置动态生成类别，保证一致性
        'names': {i: name for i, name in enumerate(config_module.CLASSES.keys())}
    }
    # 只有当测试集存在时才加入yaml
    if test_dir:
        yaml_content['test'] = test_dir

    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"动态生成数据集配置文件: {yaml_path}")
    return yaml_path


def run(config_module: Type, training_mode: str):
    """[重构] 模型训练引擎主入口，支持 'pretrain' 和 'finetune' 模式。"""
    print(f"\n--- 启动模型训练引擎 [{training_mode.upper()}] [V2.1] ---")

    trainer_cfg = config_module.TRAINER_CONFIG
    model_cfg = config_module.MODEL_CONFIG
    blueprint_name = config_module.BLUEPRINT_DIR.name
    output_models_dir = trainer_cfg["OUTPUT_MODELS_DIR"]
    
    # --- 1. 根据模式选择数据集和模型 ---
    if training_mode == 'pretrain':
        dataset_path = config_module.SYNTHESIS_CONFIG["OUTPUT_DATASET_DIR"]
        model_to_load = str(model_cfg['BASE_MODEL'])
        epochs = trainer_cfg['EPOCHS']
        batch_size = trainer_cfg['BATCH_SIZE']
        project_name = f"{blueprint_name}_pretrain"
        
    elif training_mode == 'finetune':
        dataset_path = trainer_cfg["REAL_DATASET_DIR"]
        # 微调时，我们加载预训练好的模型
        model_to_load = output_models_dir / f"{blueprint_name}_pretrain.pt"
        if not model_to_load.exists():
            print(f"[致命错误] 找不到预训练模型: {model_to_load}")
            print("请先执行 'pretrain' 步骤。")
            return
        
        finetune_params = trainer_cfg['FINETUNE_CONFIG']
        epochs = finetune_params['EPOCHS']
        batch_size = finetune_params['BATCH_SIZE']
        project_name = f"{blueprint_name}_finetune"
    else:
        print(f"[致命错误] 未知的训练模式: {training_mode}")
        return

    # --- 2. 创建数据集YAML并初始化模型 ---
    try:
        dataset_yaml_path = _create_dataset_yaml(dataset_path, config_module)
        model = YOLO(str(model_to_load))
        print(f"成功加载模型: {Path(model_to_load).name}")
    except Exception as e:
        print(f"[致命错误] 初始化失败: {e}")
        return

    # --- 3. 启动训练 ---
    print(f"\n--- 开始 {training_mode} ---")
    try:
        model.train(
            data=str(dataset_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=trainer_cfg['IMG_SIZE'],
            project=str(output_models_dir),
            name=f"{project_name}_results",
            exist_ok=True
        )
        print(f"\n✅ {training_mode.capitalize()} 成功完成！")

        # --- 4. 保存最终模型 ---
        best_model_path = output_models_dir / f"{project_name}_results/weights/best.pt"
        target_path = output_models_dir / f"{project_name}.pt"
        
        if best_model_path.exists():
            best_model_path.rename(target_path)
            print(f"  > 最佳模型已保存至: {target_path}")
        else:
            print(f"[警告] 未找到训练产出的最佳模型。")

    except Exception as e:
        print(f"\n[致命错误] 训练过程中发生错误: {e}")
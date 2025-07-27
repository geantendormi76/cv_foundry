# cv_foundry/foundry_engine/exporter.py 

from pathlib import Path
from typing import Type

from ultralytics import YOLO

def run(config_module: Type, source_model: str = 'finetune'): # 默认导出微调模型
    """ 模型导出引擎主入口。"""
    print(f"\n--- 启动模型导出引擎 (Exporting from '{source_model}') ---")
    
    blueprint_name = config_module.BLUEPRINT_DIR.name
    output_models_dir = config_module.TRAINER_CONFIG["OUTPUT_MODELS_DIR"]
    
    # 源模型路径现在是动态的
    pt_model_path = output_models_dir / f"{blueprint_name}_{source_model}.pt"
    
    if not pt_model_path.exists():
        print(f"[致命错误] 找不到已训练的模型: {pt_model_path}")
        print("请先执行 'train' 步骤来培育模型。")
        return
        
    try:
        # 加载我们自己训练好的 .pt 模型
        model = YOLO(pt_model_path)
        print(f"成功加载已训练的模型: {pt_model_path.name}")
        
        # --- 核心导出逻辑 (保持不变) ---
        # 我们继续禁用简化器以规避潜在的内存错误
        print("\n--- 开始导出为ONNX格式 (已禁用简化器以规避Bug) ---")
        
        # [路径更新] ultralytics 会自动将导出的 onnx 文件
        # 存放在与 .pt 模型相同的目录中。
        # 所以我们无需特殊指定输出路径。
        onnx_path = model.export(
            format="onnx", 
            opset=12, 
            simplify=False
        )
        print(f"\n✅ 模型成功导出！")
        
        # ultralytics 8.x 导出的路径可能是一个字符串，我们最好将其转换为Path对象
        # onnx_path 的值通常是 '_outputs/models/dino_game.onnx'
        print(f"  > ONNX模型已保存至: {Path(onnx_path).resolve()}")

    except Exception as e:
        print(f"\n[致命错误] 模型导出过程中发生错误: {e}")
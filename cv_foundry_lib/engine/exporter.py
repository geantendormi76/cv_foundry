# cv_foundry/foundry_engine/exporter.py (V2 - 绕过简化器)

from pathlib import Path
from typing import Type

from ultralytics import YOLO

def run(config_module: Type):
    """模型导出引擎的主入口。"""
    print("\n--- 启动模型导出引擎 (Exporter) [V2] ---")
    
    blueprint_name = config_module.BLUEPRINT_DIR.name
    models_dir = config_module.BASE_DIR / "models"
    
    pt_model_path = models_dir / f"{blueprint_name}.pt"
    
    if not pt_model_path.exists():
        print(f"[致命错误] 找不到已训练的模型: {pt_model_path}")
        print("请先执行 'train' 步骤来培育模型。")
        return
        
    try:
        model = YOLO(pt_model_path)
        print(f"成功加载已训练的模型: {pt_model_path.name}")
        
        # --- 核心修复 ---
        # 我们显式地告诉导出器不要运行ONNX Simplifier，
        # 因为它在当前环境下导致了内存崩溃。
        # 这会生成一个未经优化的、但完全有效的ONNX模型。
        print("\n--- 开始导出为ONNX格式 (已禁用简化器以规避Bug) ---")
        onnx_path = model.export(
            format="onnx", 
            opset=12, 
            simplify=False  # <--- 这是解决问题的关键参数
        )
        print(f"\n✅ 模型成功导出！")
        print(f"  > ONNX模型已保存至: {onnx_path}")

    except Exception as e:
        print(f"\n[致命错误] 模型导出过程中发生错误: {e}")
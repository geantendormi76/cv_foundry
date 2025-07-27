# cv_foundry/main.py (V2 - 适配新架构)

import argparse
import importlib
import sys
from pathlib import Path

# --- V2 核心改动 ---
# 将核心库目录添加到Python路径中
# 这使得我们可以使用 "from cv_foundry_lib.engine import ..." 这样的绝对导入
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
# --- 改动结束 ---


def main():
    """CV_Foundry 模型的统一铸造入口。"""
    parser = argparse.ArgumentParser(
        description="CV_Foundry: 一个蓝图驱动的微型视觉模型铸造厂。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--blueprint",
        required=True,
        help="指定要操作的蓝图名称 (e.g., dino_game)"
    )
    
    parser.add_argument(
        "--step",
        required=True,
        choices=["synthesize", "train", "export", "all"],
        help="""要执行的铸造步骤:
  - synthesize: 仅生成数据集
  - train:      仅训练模型 (需要已生成的数据集)
  - export:     仅将.pt模型导出为.onnx (需要已训练的模型)
  - all:        按顺序执行以上所有步骤"""
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*20} 正在初始化 CV_Foundry {'='*20}")
    print(f"  > 目标蓝图: {args.blueprint}")
    print(f"  > 执行步骤: {args.step}")
    
    try:
        # --- V2 核心改动 ---
        # 更新config模块的导入路径
        config_module = importlib.import_module(f"cv_foundry_lib.blueprints.{args.blueprint}.config")
        print("  > 成功加载蓝图配置。")
    except ImportError:
        print(f"\n[致命错误] 无法找到蓝图 '{args.blueprint}'。")
        print(f"请确保 'cv_foundry_lib/blueprints/{args.blueprint}/' 目录和 'config.py' 文件存在。")
        sys.exit(1)

    # --- 调度引擎模块 (V2) ---
    
    if args.step in ["synthesize", "all"]:
        try:
            # 更新引擎模块的导入路径
            from cv_foundry_lib.engine import data_synthesizer
            data_synthesizer.run(config_module)
        except Exception as e:
            print(f"\n[致命错误] 数据合成步骤失败: {e}")
            sys.exit(1)

    if args.step in ["train", "all"]:
        try:
            # 更新引擎模块的导入路径
            from cv_foundry_lib.engine import trainer
            trainer.run(config_module)
        except Exception as e:
            print(f"\n[致命错误] 模型训练步骤失败: {e}")
            sys.exit(1)

    if args.step in ["export", "all"]:
        try:
            # 更新引擎模块的导入路径
            from cv_foundry_lib.engine import exporter
            exporter.run(config_module)
        except Exception as e:
            print(f"\n[致命错误] 模型导出步骤失败: {e}")
            sys.exit(1)

    print(f"\n{'='*20} CV_Foundry 任务 '{args.step}' 执行完毕 {'='*20}")


if __name__ == "__main__":
    main()
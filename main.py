# main.py 

import argparse
import importlib
import sys
from pathlib import Path

# 将 cv_foundry_lib 添加到 Python 的模块搜索路径中
# 这允许我们无论在哪个目录下运行 main.py 都能找到我们的库
lib_path = Path(__file__).parent / "cv_foundry_lib"
sys.path.insert(0, str(lib_path))

def main():
    """CV_Foundry 框架的主入口和命令行接口。"""
    
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="CV_Foundry: 一个蓝图驱动的计算机视觉模型铸造厂。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息中的换行格式
    )
    
    parser.add_argument(
        "-b", "--blueprint",
        type=str,
        required=True,
        help="指定要使用的蓝图名称 (例如: 'dino_game')"
    )
    
    parser.add_argument(
        "-s", "--step",
        type=str,
        required=True,
        choices=['synthesize', 'pretrain', 'finetune', 'export', 'all'],
        help="""选择要执行的铸造步骤:
  - synthesize: (重新)生成合成数据
  - pretrain:   使用合成数据进行预训练
  - finetune:   使用真实数据进行微调
  - export:     导出最终的微调模型为ONNX
  - all:        执行从合成到导出的所有步骤"""
    )

    args = parser.parse_args()

    # 2. 动态加载指定的蓝图配置模块
    blueprint_name = args.blueprint
    try:
        # 动态导入类似 'blueprints.dino_game.config' 的模块
        config_module_path = f"blueprints.{blueprint_name}.config"
        config_module = importlib.import_module(config_module_path)
        print(f"✅ 成功加载蓝图: {blueprint_name}")
    except ImportError:
        print(f"[致命错误] 找不到指定的蓝图 '{blueprint_name}'。")
        print(f"  请确保 'cv_foundry_lib/blueprints/{blueprint_name}/' 目录和 config.py 文件存在。")
        sys.exit(1)

    # 3. 根据步骤参数，调用相应的引擎模块
    step = args.step
    
    if step in ['synthesize', 'all']:
        from foundry_engine import data_synthesizer
        data_synthesizer.run(config_module)

    # [新] 处理 pretrain
    if step in ['pretrain', 'all']:
        from foundry_engine import trainer
        trainer.run(config_module, training_mode='pretrain')

    # [新] 处理 finetune
    if step in ['finetune', 'all']:
        from foundry_engine import trainer
        trainer.run(config_module, training_mode='finetune')

    if step in ['export', 'all']:
        from foundry_engine import exporter
        # [修改] 导出器现在需要知道导出的源是微调模型
        # (我们需要对exporter.py做个小修改)
        exporter.run(config_module, source_model='finetune')

    print(f"\n🎉 蓝图 '{blueprint_name}' 的步骤 '{step}' 已成功执行完毕！")

if __name__ == "__main__":
    main()
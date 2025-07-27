# CV_Foundry: 计算机视觉小脑铸造厂 🦖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个“蓝图驱动”的端到端计算机视觉框架，旨在通过“合成数据预训练 + 真实数据微调”的黄金标准策略，赋能开发者快速、低成本地“自我培育”出高性能的微型视觉模型。

**本项目以Chrome Dino游戏为目标，完整地展示了如何从零开始，最终铸造出一个能在真实游戏中稳定运行的AI机器人。**

---

### 🎮 立刻体验！(即开即用)

如果你只是想立即体验我们已经训练好的Dino游戏机器人，请直接查看 **[开箱即用指南](./examples/dino_bot_example/README.md)**。

---

### 🏭 成为“铸造师”：复刻完整的模型培育流程

本指南面向希望深入理解并复刻整个“合成到真实”(Synth-to-Real)流程的开发者。我们将带你走完从数据准备到模型部署的全过程。

#### **核心理念**
我们坚信，通过程序化生成海量、多样化的合成数据进行**预训练**，再结合少量、高质量的真实数据进行**微调**，是解决现实世界中“数据瓶颈”和“领域鸿沟”问题的黄金标准。`CV_Foundry`正是这一理念的工程实现。

#### **步骤 1: 环境准备**

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/CV_Foundry_Dino_Bot.git
    cd CV_Foundry_Dino_Bot
    ```

2.  **安装依赖:** (建议在Python虚拟环境中进行)
    ```bash
    pip install -r requirements.txt
    ```

#### **步骤 2: 准备“微调”用的真实数据集**

这是整个流程中唯一需要手动介入的环节，我们已经将其高度工具化。

1.  **高频采集原始截图:**
    运行数据采集工具，它会让你选择游戏区域，然后你就可以专心玩游戏，脚本会自动高频截图。
    ```bash
    python tools/capture_tool.py
    ```
    *   所有原始截图将保存在 `_inputs/real_world_data/raw_screenshots/`。

2.  **智能过滤冗余图片:**
    运行智能过滤工具，它会使用SAD算法，从数千张原始截图中自动筛选出几十到上百张“浓缩的精华”。
    ```bash
    python tools/filter_tool.py
    ```
    *   精品数据集将保存在 `_inputs/real_world_data/filtered_for_annotation/`。

3.  **上传并进行AI辅助标注:**
    *   将 `filtered_for_annotation/` 目录中的图片上传到 [Roboflow](https://roboflow.com/) 等标注平台。
    *   **手动标注30-50张**最具多样性的“种子”图片。
    *   **训练一个临时的“标注助手”模型**。
    *   使用AI助手**自动预标注**剩余图片，你只需进行**审核和修正**。
    *   最终，以 **YOLOv8** 格式导出完整的数据集，并将其解压到 `_inputs/real_world_data/annotated_data/`。

#### **步骤 3: 启动“铸造厂”培育你自己的模型**

现在，所有原料都已备齐。回到你的项目根目录，按顺序执行以下命令：

1.  **生成合成数据 (用于预训练):**
    ```bash
    python main.py --blueprint dino_game --step synthesize
    ```

2.  **执行预训练 (建立基础认知):**
    ```bash
    python main.py --blueprint dino_game --step pretrain
    ```

3.  **执行微调 (适应真实世界):**
    ```bash
    python main.py --blueprint dino_game --step finetune
    ```

4.  **导出为ONNX (打包最终产品):**
    ```bash
    python main.py --blueprint dino_game --step export
    ```

**恭喜！** 你自己的、高性能的`dino_game_finetune.onnx`模型，现在已经出现在`_outputs/models/`目录下了！你可以将它用于你自己的游戏机器人中。

### **项目架构**

本项目采用高度模块化的四层架构：
- **编排层 (`main.py`):** 框架的总控制台。
- **核心库 (`cv_foundry_lib/`):** 包含可复用的**蓝图层**和**引擎层**。
- **应用层 (`examples/`):** 消费最终模型的示例。
- **数据层 (`_inputs/`, `_outputs/`):** 清晰分离的输入与输出。

---
感谢你的使用与贡献！
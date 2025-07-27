# Dino Game Bot - 即开即用指南 🎮

欢迎使用由 `CV_Foundry` 框架培育出的Dino游戏机器人！本项目包含一个已经训练好的、高性能的ONNX模型，让你无需任何AI背景，即可立即体验AI玩游戏。

### **准备工作**

-   你的电脑上已经安装了 Python (建议 3.10+)。
-   你的电脑是 Windows, macOS, 或 Linux 系统。

### **步骤 1: 下载并安装**

1.  **下载项目:**
    打开你的终端或命令行工具，运行以下命令：
    ```bash
    git clone https://github.com/geantendormi76/cv_foundry.git
    cd cv_foundry
    ```

2.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

### **步骤 2: 启动机器人！**

1.  **打开游戏:**
    在你的Chrome浏览器中，打开Dino游戏页面: `chrome://dino`

2.  **运行脚本:**
    在你的终端中，确保当前路径在`CV_Foundry_Dino_Bot`项目下，然后运行：
    ```bash
    python examples/dino_bot_example/play_game.py
    ```

3.  **按照提示操作:**
    *   脚本会弹出一个窗口，请用鼠标**拖动框选**出完整的Dino游戏区域，然后按`ENTER`键。
    *   选择完毕后，你有**3秒钟**的时间，**用鼠标点击一下Chrome的游戏窗口**，使其成为活动窗口。
    *   然后，尽情欣赏AI为你表演吧！

### **如何退出？**

在机器人运行过程中，你可以随时点击那个名为 `CV_Foundry Dino_Bot - DEBUG VIEW` 的调试窗口，然后按下键盘上的 `q` 键来安全退出。

### **高级玩法 (可选)**

如果你觉得机器人跳得太早或太晚，可以尝试修改 `controller.py` 文件中的 `REACTION_DISTANCE_THRESHOLD` 参数，来调整它的反应灵敏度。

---
享受游戏吧！
# cv_foundry/utils/keyboard_controller.py

import time
from pynput.keyboard import Controller, Key

class KeyboardController:
    """一个用于模拟键盘按键的简单工具类。"""
    
    def __init__(self):
        self.keyboard = Controller()

    def jump(self, duration=0.1):
        """模拟一次跳跃动作（按下并快速松开空格键）。"""
        self.keyboard.press(Key.space)
        time.sleep(duration)
        self.keyboard.release(Key.space)

    def duck(self, duration=0.3):
        """模拟一次下蹲动作（按下并松开下箭头键）。"""
        self.keyboard.press(Key.down)
        time.sleep(duration)
        self.keyboard.release(Key.down)
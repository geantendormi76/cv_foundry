# cv_foundry/utils/screen_capture.py (V2 - 使用OpenCV)

import cv2
import mss
import numpy as np

class ScreenCapturer:
    """一个用于选择屏幕区域并进行高效截图的工具类。"""
    
    def __init__(self):
        self.roi = None
        self.sct = mss.mss()

    def select_roi(self):
        """显示全屏截图，让用户使用OpenCV的工具选择游戏区域。"""
        print("\n准备选择区域... 将会截取您的主屏幕。")
        
        # 1. 截取整个主屏幕
        monitor = self.sct.monitors[1] # monitor[0]是所有屏幕的合集，[1]是主屏幕
        full_screenshot = np.array(self.sct.grab(monitor))
        full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)

        # 2. 使用OpenCV的selectROI
        window_name = "请用鼠标拖动选择Dino游戏区域, 然后按 ENTER 或 SPACE 键确认"
        print(f"在弹出的 '{window_name}' 窗口中操作...")
        roi_coords = cv2.selectROI(window_name, full_screenshot_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        if sum(roi_coords) == 0: # 用户按ESC取消
            print("[错误] 未选择任何区域，程序退出。")
            exit()
            
        x, y, w, h = roi_coords
        
        # 3. 将相对坐标转换为绝对屏幕坐标
        self.roi = {
            "top": monitor["top"] + y, 
            "left": monitor["left"] + x, 
            "width": w, 
            "height": h
        }
        print(f"✅ 游戏区域选择成功: {self.roi}")
            
    def capture(self) -> np.ndarray:
        """捕获已选定区域的屏幕截图。"""
        if not self.roi:
            raise ValueError("必须先调用 select_roi() 来选择一个区域。")
        
        sct_img = self.sct.grab(self.roi)
        img = np.array(sct_img)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
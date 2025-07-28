# tools/capture_tool.py (Windows版)

import time
from pathlib import Path
import cv2
import mss
import numpy as np
from pynput import keyboard

# [路径修改] 输出到用户桌面下的一个新文件夹，非常直观
OUTPUT_DIR = Path.home() / "Desktop" / "cv_foundry_capture" / "raw_screenshots"
CAPTURE_INTERVAL_SECONDS = 0.2
STOP_KEY = keyboard.Key.esc

# ... (其余代码与之前的WSL版本完全相同) ...
capturing = True
roi = None

def on_press(key):
    global capturing
    if key == STOP_KEY:
        print(f"\n检测到'{STOP_KEY}'被按下，正在停止采集...")
        capturing = False
        return False

def select_roi(sct):
    print("准备选择区域... 将会截取您的主屏幕。")
    monitor = sct.monitors[1]
    full_screenshot = np.array(sct.grab(monitor))
    full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)
    window_name = "请拖动选择游戏区域, 然后按 ENTER 确认"
    roi_coords = cv2.selectROI(window_name, full_screenshot_bgr, fromCenter=False)
    cv2.destroyWindow(window_name)
    if sum(roi_coords) == 0: return None
    x, y, w, h = roi_coords
    return {"top": monitor["top"] + y, "left": monitor["left"] + x, "width": w, "height": h}

def main():
    global roi
    print("--- CV_Foundry 真实数据采集工具 (Windows版) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        roi = select_roi(sct)
        if roi is None:
            print("[错误] 未选择任何区域，程序退出。")
            return
        print(f"✅ 区域选择成功: {roi}")
        print(f"将在3秒后开始高频截图，请切换到游戏窗口...")
        print(f"采集过程中，按 'ESC' 键可随时停止。")
        time.sleep(3)
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        count = 0
        while capturing:
            sct_img = sct.grab(roi)
            img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            timestamp = time.time_ns()
            cv2.imwrite(str(OUTPUT_DIR / f"{timestamp}.png"), img_bgr)
            count += 1
            print(f"已采集 {count} 帧...", end='\r')
            time.sleep(CAPTURE_INTERVAL_SECONDS)
    print(f"\n采集结束。共采集 {count} 帧到目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
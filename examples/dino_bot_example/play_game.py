# play_game.py (黄金标准 V7.1 - 最终修正版)

import cv2
import numpy as np
import time
from pathlib import Path
import sys
from ultralytics import YOLO

example_dir = Path(__file__).resolve().parent
sys.path.append(str(example_dir))

from utils.keyboard_controller import KeyboardController
from utils.screen_capture import ScreenCapturer
from controller import get_action

def run_bot(model_path: Path):
    """最终版机器人主函数 - 采用智能冷却逻辑"""
    print("--- 启动Dino游戏机器人 (V7.2 - 智能冷却版) ---")
    
    try:
        model = YOLO(model_path, task='detect')
        print(f"✅ 模型通过Ultralytics加载成功: {model_path.name}")
    except Exception as e:
        print(f"[致命错误] 模型加载失败: {e}")
        return
        
    class_map = {0: 'bird', 1: 'cactus', 2: 'dino'} 
    print(f"   > 识别类别: {class_map}")

    capturer = ScreenCapturer()
    keyboard = KeyboardController()
    capturer.select_roi()
    if capturer.roi is None: return 
    print("\n3秒后机器人将开始运行...")
    time.sleep(3)
    
    # [新] 初始化动作冷却相关变量
    last_action_time = 0  # 记录上次执行动作的时间戳
    ACTION_COOLDOWN = 0.4 # 设置一个0.4秒的冷却时间，防止按键连发

    print("\n🤖 AI已接管！按 'q' 键退出机器人。")
    while True:
        frame = capturer.capture()
        if frame is None: continue
        
        results = model(frame, verbose=False, conf=0.45) 
        
        result = results[0]
        boxes = result.boxes
        frame_h, frame_w, _ = frame.shape  

        action = get_action(boxes, class_map, frame_h)
        
        # [核心修改] 引入智能冷却判断
        current_time = time.time()
        if action and (current_time - last_action_time > ACTION_COOLDOWN):
            if action == "jump":
                keyboard.jump()
                print(f">>> JUMP! (检测到 {len(boxes)} 个物体)")
            elif action == "duck":
                keyboard.duck()  
                print(f">>> DUCK! (检测到 {len(boxes)} 个物体)")
            
            last_action_time = current_time # 执行动作后，更新时间戳
        
        debug_frame = result.plot(conf=True)
        
        cv2.imshow("CV_Foundry Dino_Bot - DEBUG VIEW", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    onnx_model_path = project_root / "models" / "dino_game_finetune.onnx"
    
    if not onnx_model_path.exists():
        print(f"[致命错误] 找不到ONNX模型文件: {onnx_model_path}")
    else:
        run_bot(onnx_model_path)
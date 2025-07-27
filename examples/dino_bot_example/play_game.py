# cv_foundry/play_game.py (V5 - 基于相对距离决策)

import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
import sys

# 将项目根目录添加到Python的系统路径中
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from utils.keyboard_controller import KeyboardController
from utils.screen_capture import ScreenCapturer

def run_bot(onnx_path: Path, conf_threshold: float = 0.65):
    """运行Dino游戏机器人的主函数。"""
    print("--- 启动Dino游戏机器人 (V5 - 终极版) ---")
    
    # --- 1. 初始化 ---
    session = ort.InferenceSession(str(onnx_path))
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_name = model_inputs[0].name
    
    # V5: 从ONNX模型元数据中动态获取类别名称
    class_names_str = session.get_modelmeta().custom_metadata_map['names']
    # 解析字符串 '{\'0\': \'cactus\', \'1\': \'dino\'}'
    class_map = eval(class_names_str)
    print(f"✅ ONNX模型加载成功: {onnx_path.name} | 识别类别: {class_map}")

    capturer = ScreenCapturer()
    keyboard = KeyboardController()

    # --- 2. 选择游戏区域 ---
    capturer.select_roi()
    
    print("\n准备就绪！请将鼠标点击到Dino游戏窗口，准备开始...")
    print("3秒后机器人将开始运行...")
    time.sleep(3)
    
    print("\n🤖 AI已接管！按 'q' 键退出机器人。")
    while True:
        # a. 截图
        frame = capturer.capture()
        if frame is None: continue
        
        # b. 预处理
        _, _, height, width = input_shape
        input_img = cv2.resize(frame, (width, height))
        input_img = input_img.transpose(2, 0, 1)
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32) / 255.0

        # c. 推理
        results = session.run(None, {input_name: input_img})

        # d. 后处理与决策 (全新逻辑)
        detections = results[0][0].T
        
        dino_pos = None
        closest_cactus_pos = None
        
        debug_frame = frame.copy()

        for detection in detections:
            confidence = detection[4]
            # V5: 从检测结果中获取类别ID
            class_id = int(detection[5]) 
            
            if confidence >= conf_threshold:
                label = class_map.get(class_id, 'unknown')
                cx_norm = detection[0] # 归一化的中心X坐标
                
                if label == 'dino':
                    dino_pos = cx_norm
                elif label == 'cactus':
                    # 只考虑在恐龙前方的仙人掌
                    if dino_pos is None or cx_norm > dino_pos:
                        if closest_cactus_pos is None or cx_norm < closest_cactus_pos:
                            closest_cactus_pos = cx_norm
                
                # 绘制所有检测框
                _, cy_norm, w_norm, h_norm = detection[:4]
                frame_h, frame_w = debug_frame.shape[:2]
                x1 = int((cx_norm - w_norm / 2) * frame_w)
                y1 = int((cy_norm - h_norm / 2) * frame_h)
                x2 = int((cx_norm + w_norm / 2) * frame_w)
                y2 = int((cy_norm + h_norm / 2) * frame_h)
                color = (0, 255, 0) if label == 'cactus' else (255, 0, 0) # 仙人掌绿色，恐龙红色
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_frame, f"{label}:{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # e. V5 核心决策逻辑
        if dino_pos is not None and closest_cactus_pos is not None:
            relative_distance = closest_cactus_pos - dino_pos
            log_message = f"相对距离: {relative_distance:.3f}"
            print(log_message, end='\r')
            
            # 如果仙人掌在恐龙前方，并且相对距离小于某个反应阈值
            if relative_distance > 0 and relative_distance < 0.35: # 0.35这个阈值可以根据实际情况微调
                keyboard.jump()
                print(f"\n>>> JUMPING! 触发距离: {relative_distance:.3f} <<<\n")
                time.sleep(0.4) # 跳跃后增加冷却时间，避免因同一障碍物连续触发跳跃
        else:
            print("等待同时检测到恐龙和仙人掌...", end='\r')


        cv2.imshow("CV_Foundry Dino_Bot - DEBUG VIEW (V5)", debug_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n程序退出。")
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    onnx_model_path = Path(__file__).resolve().parent / "models/dino_game.onnx"
    
    if not onnx_model_path.exists():
        print(f"[致命错误] 找不到ONNX模型文件: {onnx_model_path}")
    else:
        run_bot(onnx_model_path)
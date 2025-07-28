# tools/filter_tool.py (Windows版)

import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# [路径修改] 从用户桌面读取和写入
BASE_DIR = Path.home() / "Desktop" / "cv_foundry_capture"
INPUT_DIR = BASE_DIR / "raw_screenshots/"
OUTPUT_DIR = BASE_DIR / "filtered_for_annotation/"
SAD_THRESHOLD_PER_PIXEL = 1.5


def calculate_sad(img1, img2):
    if img1.shape != img2.shape: return float('inf') 
    return np.sum(np.abs(img1.astype("float") - img2.astype("float")))

def main():
    print("--- CV_Foundry 冗余数据智能过滤工具 (SAD黄金标准版) ---")
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"[错误] 输入目录 '{INPUT_DIR}' 不存在或为空。")
        print("请先运行 'capture_tool.py' 采集原始截图。")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_files = sorted(os.listdir(INPUT_DIR))
    if not image_files:
        print("[警告] 输入目录中没有找到图片。")
        return
    last_accepted_image = None
    accepted_count = 0
    print(f"正在从 {len(image_files)} 张原始图片中过滤精华...")
    for filename in tqdm(image_files, desc="过滤进度"):
        if not filename.endswith(('.png', '.jpg', '.jpeg')): continue
        img_path = INPUT_DIR / filename
        current_image = cv2.imread(str(img_path))
        if last_accepted_image is None:
            cv2.imwrite(str(OUTPUT_DIR / filename), current_image)
            last_accepted_image = current_image
            accepted_count += 1
            continue
        sad = calculate_sad(last_accepted_image, current_image)
        height, width, _ = current_image.shape
        sad_per_pixel = sad / (height * width * 3)
        if sad_per_pixel > SAD_THRESHOLD_PER_PIXEL:
            cv2.imwrite(str(OUTPUT_DIR / filename), current_image)
            last_accepted_image = current_image
            accepted_count += 1
    print("\n✅ 过滤完成！")
    print(f"  > 共处理 {len(image_files)} 张原始图片。")
    print(f"  > 筛选出 {accepted_count} 张具有显著变化的图片。")
    print(f"  > 精品数据集已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
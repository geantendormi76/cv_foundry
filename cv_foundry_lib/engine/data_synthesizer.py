# cv_foundry/foundry_engine/data_synthesizer.py (V2 - 修复版)

import os
import random
from pathlib import Path
from typing import Type

from PIL import Image, ImageChops
from tqdm import tqdm

def _create_mask_from_green_screen(img: Image.Image) -> Image.Image:
    """从绿幕图像创建蒙版。"""
    # 找到所有非绿色的像素点。我们给一个容差范围。
    green_min = (0, 200, 0, 255)
    green_max = (100, 255, 100, 255)
    
    # 创建一个与图像大小相同的黑色蒙版
    mask = Image.new('L', img.size, 0)
    
    # 遍历所有像素
    for x in range(img.width):
        for y in range(img.height):
            p = img.getpixel((x, y))
            # 如果像素不在绿色范围内，就在蒙版对应位置画一个白色点
            if not (green_min[0] <= p[0] <= green_max[0] and 
                    green_min[1] <= p[1] <= green_max[1] and 
                    green_min[2] <= p[2] <= green_max[2]):
                mask.putpixel((x, y), 255) # 255代表不透明
    return mask

def _get_asset_images(config_module: Type) -> dict:
    """加载所有视觉资产并移除绿幕。"""
    assets_path = config_module.ASSETS_PATH
    asset_images = {cls: [] for cls in config_module.CLASSES.keys()}

    for asset_file in assets_path.glob("*.png"):
        # V2修复：使用'-'来分割，获取类别名
        class_name = asset_file.name.split("-")[0]
        if class_name in asset_images:
            try:
                img_rgba = Image.open(asset_file).convert("RGBA")
                
                # V2新增：创建蒙版并将其作为alpha通道应用
                mask = _create_mask_from_green_screen(img_rgba)
                img_rgba.putalpha(mask)

                asset_images[class_name].append(img_rgba)
            except Exception as e:
                print(f"[警告] 无法加载或处理资产 '{asset_file.name}': {e}")
    
    loaded_count = sum(len(v) for v in asset_images.values())
    if loaded_count == 0:
        print("[严重错误] 未能成功加载任何有效资产！请检查文件名和文件内容。")
    else:
        print(f"成功加载并处理了 {loaded_count} 个视觉资产。")
    return asset_images

def _generate_dataset(num_images: int, output_dir: Path, config_module: Type, asset_images: dict):
    """生成指定数量的图片和标签。"""
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cfg = config_module.SYNTHESIS_CONFIG
    canvas_w, canvas_h = cfg["IMAGE_WIDTH"], cfg["IMAGE_HEIGHT"]

    print(f"开始生成 {num_images} 张图片到 '{output_dir.name}' 目录...")
    for i in tqdm(range(num_images), desc=f"合成 {output_dir.name} 数据集"):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        annotations = []
        
        num_obstacles = random.randint(1, cfg["MAX_OBSTACLES_PER_IMAGE"])

        # 防止在没有加载到任何资产的情况下进入死循环
        if not any(asset_images.values()):
            continue
            
        for _ in range(num_obstacles):
            class_name = random.choice(list(config_module.CLASSES.keys()))
            if not asset_images.get(class_name):
                continue
            
            asset_rgba = random.choice(asset_images[class_name]).copy()
            scale = random.uniform(*cfg["SCALE_RANGE"])
            new_w = int(asset_rgba.width * scale)
            new_h = int(asset_rgba.height * scale)
            asset_resized = asset_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            paste_x = random.randint(0, canvas_w - new_w)
            paste_y = canvas_h - new_h
            
            # V2修复：粘贴时使用资产自身的alpha通道作为蒙版
            canvas.paste(asset_resized, (paste_x, paste_y), asset_resized)

            class_id = config_module.CLASSES[class_name]
            center_x = paste_x + new_w / 2
            center_y = paste_y + new_h / 2
            norm_cx = center_x / canvas_w
            norm_cy = center_y / canvas_h
            norm_w = new_w / canvas_w
            norm_h = new_h / canvas_h
            annotations.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        if annotations:
            img_path = img_dir / f"synth_{i}.png"
            lbl_path = lbl_dir / f"synth_{i}.txt"
            canvas.save(img_path)
            with open(lbl_path, "w") as f:
                f.write("\n".join(annotations))

# 'run' 函数保持不变
def run(config_module: Type):
    """数据合成引擎的主入口。"""
    print("\n--- 启动数据合成引擎 (Data Synthesizer) [V2] ---")
    asset_images = _get_asset_images(config_module)
    blueprint_name = config_module.BLUEPRINT_DIR.name
    dataset_root = config_module.BASE_DIR / "datasets" / blueprint_name
    cfg = config_module.SYNTHESIS_CONFIG
    _generate_dataset(cfg["NUM_TRAIN_IMAGES"], dataset_root / "train", config_module, asset_images)
    _generate_dataset(cfg["NUM_VAL_IMAGES"], dataset_root / "val", config_module, asset_images)
    print("\n✅ 数据合成引擎运行完毕！")
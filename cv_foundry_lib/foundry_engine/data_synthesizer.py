# cv_foundry/foundry_engine/data_synthesizer.py (黄金标准 V2.0 - 重构版)

import os
import random
from pathlib import Path
from typing import Type

from PIL import Image, ImageChops
from tqdm import tqdm


def _create_mask_from_single_color_bg(img: Image.Image, tolerance: int = 25) -> Image.Image:
    """从单一颜色背景的图像创建蒙版（对绿幕、灰幕、黑幕都有效）。"""
    # 将图像转换为带Alpha通道的格式，以防万一
    img_rgba = img.convert("RGBA")
    
    # 自动识别背景色：我们假设左上角第一个像素就是背景色
    bg_pixel = img_rgba.getpixel((0, 0))
    
    # 创建一个与图像大小相同的全透明蒙版
    mask = Image.new('L', img_rgba.size, 0)
    
    # 遍历所有像素
    for x in range(img_rgba.width):
        for y in range(img_rgba.height):
            p = img_rgba.getpixel((x, y))
            
            # 计算当前像素与背景色的差异
            is_background = all(abs(p[i] - bg_pixel[i]) <= tolerance for i in range(3))

            # 如果像素不是背景色，就在蒙版对应位置画一个不透明的点
            if not is_background:
                mask.putpixel((x, y), 255) # 255代表不透明
    return mask


def _get_asset_images(config_module: Type) -> dict:
    """加载所有视觉资产并移除背景。"""
    assets_path = config_module.ASSETS_PATH
    asset_images = {cls: [] for cls in config_module.CLASSES.keys()}

    for asset_file in assets_path.glob("*.png"):
        class_name = asset_file.name.split("-")[0]
        if class_name in asset_images:
            try:
                img_rgba = Image.open(asset_file).convert("RGBA")
                
                # [修改] 调用我们全新的、更智能的蒙版创建函数
                mask = _create_mask_from_single_color_bg(img_rgba)
                
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
    """
    生成指定数量的图片和标签到指定的输出目录。
    """
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
        
        # [逻辑简化] 移除原有的FORCE_DINO的复杂逻辑，我们可以在更高层次保证
        # 或者在循环内部进行更简单的处理。
        # 此处为了清晰，我们暂时简化为随机生成。
        
        num_obstacles = random.randint(1, cfg["MAX_OBSTACLES_PER_IMAGE"])
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
            # 简化逻辑，假设所有物体都在地面上
            paste_y = canvas_h - new_h
            
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

# run 函数需要重大重构，以使用新的配置
def run(config_module: Type):
    """[重构] 数据合成引擎的主入口。"""
    print("\n--- 启动数据合成引擎 (Data Synthesizer)  ---")
    
    asset_images = _get_asset_images(config_module)
    
    # 从配置中获取所有需要的信息
    cfg = config_module.SYNTHESIS_CONFIG
    dataset_root = cfg["OUTPUT_DATASET_DIR"]
    
    # 为训练集和验证集调用生成函数
    _generate_dataset(
        num_images=cfg["NUM_TRAIN_IMAGES"], 
        output_dir=dataset_root / "train", 
        config_module=config_module, 
        asset_images=asset_images
    )
    
    _generate_dataset(
        num_images=cfg["NUM_VAL_IMAGES"], 
        output_dir=dataset_root / "val", 
        config_module=config_module, 
        asset_images=asset_images
    )
    
    print(f"\n✅ 数据合成引擎运行完毕！数据集已生成至 '{dataset_root}'")
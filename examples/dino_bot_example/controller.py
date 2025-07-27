# controller.py (黄金标准最终版 V3.1 - 容错边距决策)

def get_action(boxes, class_map, frame_h):
    """
    根据障碍物与恐龙的动态相对位置和相对高度来决策。
    增加了Y轴判断的容错边距，以应对临界情况。
    """
    dino_box = None
    obstacles = []
    
    for box in boxes:
        class_id = int(box.cls[0])
        label = class_map.get(class_id, 'unknown')
        if label == 'dino':
            dino_box = box
        elif label in ['cactus', 'bird']:
            obstacles.append(box)
            
    if not dino_box or not obstacles:
        return None
        
    dino_x_center = (dino_box.xyxy[0].numpy()[0] + dino_box.xyxy[0].numpy()[2]) / 2
    
    closest_obstacle = None
    min_distance = float('inf')
    
    for obstacle in obstacles:
        obs_coords = obstacle.xyxy[0].numpy()
        obs_x_center = (obs_coords[0] + obs_coords[2]) / 2
        distance = obs_x_center - dino_x_center
        if 0 < distance < min_distance:
            min_distance = distance
            closest_obstacle = obstacle

    if not closest_obstacle:
        return None
        
    # 在智能冷却模式下，这个值可以适当调小一点，反应会更极限
    REACTION_DISTANCE_THRESHOLD = 130 
    
    if min_distance < REACTION_DISTANCE_THRESHOLD:
        closest_obs_id = int(closest_obstacle.cls[0])
        closest_obs_label = class_map.get(closest_obs_id, 'unknown')
        
        print(f">>> 危险! 最近的 {closest_obs_label} 距离仅剩 {min_distance:.2f} 像素! <<<")

        if closest_obs_label == 'cactus':
            return "jump"
        
        elif closest_obs_label == 'bird':
            dino_y_center = (dino_box.xyxy[0].numpy()[1] + dino_box.xyxy[0].numpy()[3]) / 2
            closest_obs_y_center = (closest_obstacle.xyxy[0].numpy()[1] + closest_obstacle.xyxy[0].numpy()[3]) / 2
            
            # [核心逻辑修正 V3.1]
            # 引入一个Y轴的容错边距。因为恐龙检测框较高，其中心点也高。
            # 即使小鸟中心点稍微高于恐龙中心点，也可能构成威胁。
            # 这个值意味着，只要小鸟的Y中心不比恐龙的Y中心高出15个像素，都判定为低飞/中飞，需要跳跃。
            Y_TOLERANCE = 15 # 可以根据实际情况微调
            
            print(f"   > 决策依据: 鸟Y中心({closest_obs_y_center:.2f}) vs 龙Y危险区上限({dino_y_center - Y_TOLERANCE:.2f})")

            # 新逻辑：如果小鸟的Y中心 > (恐龙的Y中心 - 容错值)，就跳
            if closest_obs_y_center > (dino_y_center - Y_TOLERANCE):
                return "jump" # 应对低飞和中飞小鸟
            else:
                return "duck" # 仅应对真正的高飞小鸟
                
    return None
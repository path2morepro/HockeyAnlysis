import math

# 定义冰球场地上的关键位置
def position():
    return {
        "rink_length": 200,
        "rink_width": 85,
        "goal_line_x": 89,
        "goal_center": (89, 0),
        "goal_width": 6,
        "goal_top": (89, 3),
        "goal_bottom": (89, -3),
        "danger_zone_radius": 4,
        "faceoff_radius": 15,
        "faceoff_spots": {
            "Q1": (69, 22),
            "Q2": (-69, 22),
            "Q3": (-69, -22),
            "Q4": (69, -22),
        },
        "blue_line_x": [25, -25],
    }

# 判断一个点是否在危险区（球门前的半圆区域）
def is_in_danger_zone(x, y):
    pos = position()
    gx, gy = pos["goal_center"]
    r = pos["danger_zone_radius"]
    distance = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    return distance <= r and x <= gx  # 半圆面朝球场中间

# 判断一个点是否在slot区域
def is_in_slot(x, y):
    pos = position()
    # 利用绝对值
    # 仅一次判断是否位于半个slot区内
    x = abs(x)
    y = abs(y)
    gx, gy = pos["goal_center"]
    gt_y_top = pos["goal_top"][1]
    gt_y_bot = pos["goal_bottom"][1]
    faceoff_x, faceoff_y = pos["faceoff_spots"]["Q1"]
    faceoff_radius = pos['faceoff_radius']

    # 分区判断y
    # slot在faceoff圆心和球门线之间是梯形
    # slot在faceoff圆心和faceoff spot区边界是矩形
    if faceoff_x - faceoff_radius <= x <= gx:
        if gy <= y <= (faceoff_y if x <= faceoff_x else -0.95 * x + 87.55):
            return True
    return False
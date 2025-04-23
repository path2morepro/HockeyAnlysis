import math
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
import pandas as pd

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
        "blue_line_x": 25,
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

def inner_blue_line(x):
    pos = position
    blue_line = pos['blue_line_x']
    return True if x > blue_line else False

def aggregate_stats(data_A, data_B):
    """
    将射门数据和犯规数据按照teamid和gameid组合进行聚合
    
    参数:
        data_A: (含teamid和gameid列)
        penalty_data: 犯规数据 DataFrame (含teamid和gameid列)
    
    返回:
        aggregated_stats: 聚合后的数据 DataFrame
    """
    # 确保两个DataFrame都有teamid和gameid列
    if not all(col in data_A.columns for col in ['teamid', 'gameid']):
        raise ValueError("data_A 必须包含 teamid 和 gameid 列")
    
    if not all(col in data_B.columns for col in ['teamid', 'gameid']):
        raise ValueError("data_B 必须包含 teamid 和 gameid 列")
    
    # 基于teamid和gameid进行合并
    aggregated_stats = data_A.merge(
        data_B,
        on=['teamid', 'gameid'],
        how='outer',  # 使用outer join以保留所有记录
        suffixes=('_shots', '_penalty')
    )
    
    # 填充可能的空值
    aggregated_stats = aggregated_stats.fillna(0)
    
    return aggregated_stats


def analysis(agg_df, features):
    # 9. Filter unique matchups to ensure one per game and directionality
    df_matches = agg_df[['gameid', 'teamid', 'style', 'result']].copy()
    df_matches = df_matches.merge(df_matches, on='gameid')
    df_matches = df_matches[df_matches['teamid_x'] < df_matches['teamid_y']].copy()

    # 10. Compute win rate by style vs style
    records = []
    for _, row in df_matches.iterrows():
        style1, result1 = row['style_x'], row['result_x']
        style2, result2 = row['style_y'], row['result_y']
        if result1 == 1:
            records.append((style1, style2, 1))
            records.append((style2, style1, 0))
        else:
            records.append((style1, style2, 0))
            records.append((style2, style1, 1))

    results_df = pd.DataFrame(records, columns=['style_team', 'style_opp', 'win'])
    winrate_table = results_df.groupby(['style_team', 'style_opp'])['win'].mean().unstack().round(2)
    print("\nSymmetric Win Rate Matrix (Sum = 1 per matchup pair):")
    print(winrate_table)

    # 11. Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(winrate_table, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Win Rate by Team Style vs Opponent Style (Symmetric)")
    plt.xlabel("Opponent Style")
    plt.ylabel("Team Style")
    plt.tight_layout()
    plt.show()

    # 12. Radar chart to visualize style features
    style_features = agg_df.groupby('style')[features].mean()
    style_features_norm = (style_features - style_features.min()) / (style_features.max() - style_features.min())

    labels = style_features.columns.tolist()
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("Set2", n_colors=3)

    for idx, (style, row) in enumerate(style_features_norm.iterrows()):
        values = row.tolist()
        values += values[:1]
        plt.polar(angles, values, label=style, color=colors[idx])
        plt.fill(angles, values, alpha=0.1, color=colors[idx])

    plt.xticks(angles[:-1], labels, fontsize=8)
    plt.title("Team Style Feature Radar Chart", size=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
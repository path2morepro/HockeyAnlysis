import math
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


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


'''
给出参数：gameid, teamid, agg_df, possession_changes
返回gameid, teamid, stype(from agg_df),  xadjcoord, yadjcoord (the last 2 from possession changes)
要求按照gameid, teamid在agg_df中找到style，将该行(gameid, teamid, style)再根据possession_changes中的2个id返回x,y
用x,y画出热力图，并标注gameid, teamid, style和x=0, x=25, x=-25三条线
''' 
# call case
# visualize_possession_changes(69989, 726, agg_df, possession_changes)
# visualize_possession_changes(80975, 503, agg_df, possession_changes)
def visualize_possession_changes(gameid, teamid, agg_df, possession_changes):
    rink = position()

    # Get team style
    team_style = agg_df[(agg_df['gameid'] == gameid) & (agg_df['teamid'] == teamid)]
    style = team_style['style'].iloc[0] if 'style' in team_style.columns and not team_style.empty else "Unknown style"

    # Filter possession changes
    team_possession_changes = possession_changes[(possession_changes['gameid'] == gameid) & (possession_changes['teamid'] == teamid)]
    if team_possession_changes.empty:
        print(f"No possession changes found for team {teamid} in game {gameid}")
        return

    x_coords = team_possession_changes['xadjcoord'].values
    y_coords = team_possession_changes['yadjcoord'].values

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Rink outline
    ax.plot([-100, 100, 100, -100, -100], [-42.5, -42.5, 42.5, 42.5, -42.5], 'k-', linewidth=2)

    # Center and blue lines
    ax.axvline(x=0, color='r', linestyle='-', linewidth=1.5)
    ax.axvline(x=rink["blue_line_x"], color='b', linestyle='-', linewidth=1.5)
    ax.axvline(x=-rink["blue_line_x"], color='b', linestyle='-', linewidth=1.5)

    # Center faceoff circle
    center_circle = Circle((0, 0), rink["faceoff_radius"], fill=False, color='b', linewidth=1)
    ax.add_patch(center_circle)

    # Faceoff spots and circles
    for name, spot in rink["faceoff_spots"].items():
        circle = Circle(spot, rink["faceoff_radius"], fill=False, color='b', linestyle='--', linewidth=1)
        dot = Circle(spot, 0.8, color='blue')
        ax.add_patch(circle)
        ax.add_patch(dot)

    # Goal areas
    for x in [-rink["goal_line_x"], rink["goal_line_x"]]:
        ax.plot([x, x], [-rink["goal_width"] / 2, rink["goal_width"] / 2], 'k-', linewidth=3)
        goal_box = Rectangle((x, -3), 1, 6, color='gray', alpha=0.3)
        ax.add_patch(goal_box)

    # Heatmap
    if len(x_coords) > 5:
        sns.kdeplot(x=x_coords, y=y_coords, cmap="YlOrRd", fill=True, bw_adjust=0.7, ax=ax)
    else:
        ax.scatter(x_coords, y_coords, c='red', alpha=0.7, s=50)

    # Points
    ax.scatter(x_coords, y_coords, c='black', alpha=0.3, s=10)

    # Labels and decorations
    ax.set_title(f"Possession Changes - Game {gameid}, Team {teamid}, Style: {style}")
    ax.set_xlabel("X Coordinate (ft)")
    ax.set_ylabel("Y Coordinate (ft)")
    ax.legend(handles=[
        Line2D([0], [0], color='r', lw=1.5, label='Center Line (x=0)'),
        Line2D([0], [0], color='b', lw=1.5, label='Blue Lines (x=±25)'),
    ])
    ax.set_xlim(-105, 105)
    ax.set_ylim(-45, 45)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# 返回所有球权交换事件的df['eventname']和df['type']
# 球权交换事件定义df['teaminpossession'][i] != df['teaminpossession'][i+1] 
def possession_change_events(df):
    df = df.dropna(subset=['teaminpossession'])
    """
    返回所有球权交换事件的数据+teamid的比赛风格+坐标信息
    参数:
        df: 原始数据 DataFrame，必须包含 'teaminpossession', 'eventname', 和 'type' 列
    
    返回:
        possession_changes: 包含所有球权交换事件的 DataFrame
    """
    # # 确保数据按时间顺序排序
    # if 'compiledgametime' in df.columns:
    #     df = df.sort_values(by=['gameid', 'compiledgametime']).reset_index(drop=True)
    
    # 找出球权交换的位置
    # 比较当前行和下一行的teaminpossession
    possession_changes = df[:-1][df['teaminpossession'][:-1].values != df['teaminpossession'][1:].values]
    possession_changes = possession_changes[possession_changes['eventname'] != 'faceoff']
    # 也可以包含下一行的信息（即导致球权交换的事件）
    next_events = df[1:][df['teaminpossession'][:-1].values != df['teaminpossession'][1:].values].reset_index(drop=True)
    
    # 将球权交换事件和导致球权交换的下一个事件合并
    possession_changes = possession_changes.reset_index(drop=True)
    # possession_changes['next_eventname'] = next_events['eventname']
    # possession_changes['next_type'] = next_events['type'] if 'type' in next_events.columns else None
    possession_changes['next_teaminpossession'] = next_events['teaminpossession']
    
    # 选择需要的列
    result_columns = ['gameid', 'teamid', 'compiledgametime',
                      'teaminpossession', 'eventname', 'type', 
                      'next_teaminpossession', 'xadjcoord', 'yadjcoord']
    
    # # 确保所有列都存在
    # result_columns = [col for col in result_columns if col in possession_changes.columns]
    
    return possession_changes[result_columns]

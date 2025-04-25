import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def possession_change_events(df, teamid_filter=None):
    """
    返回所有丢失球权事件，基于 ishomegame 和 period 统一转换坐标到“向右进攻”视角。

    参数:
        df: 原始事件数据（必须包含 teamid, teaminpossession, xadjcoord, yadjcoord, ishomegame, period）
        teamid_filter: 若提供，仅返回该 team 的事件

    返回:
        DataFrame: 球权丢失事件 + 坐标统一到右攻视角
    """
    df = df.dropna(subset=['teaminpossession', 'ishomegame'])
    df = df.sort_values(by=['gameid', 'compiledgametime']).reset_index(drop=True)

    # 当前和下一行控球情况
    current_poss = df['teaminpossession'].values[:-1]
    next_poss = df['teaminpossession'].values[1:]
    current_team = df['teamid'].values[:-1]

    # 丢球事件（本队控球 -> 非本队控球）
    lost_mask = (current_poss == current_team) & (next_poss != current_team)
    loss_df = df[:-1][lost_mask].copy()

    # 获取 ishomegame 和 period
    loss_df['ishomegame'] = df['ishomegame'].values[:-1][lost_mask]
    loss_df['period'] = df['period'].values[:-1][lost_mask]
    loss_df['next_teaminpossession'] = next_poss[lost_mask]

    # 计算方向：如果当前 period 是主队向右（1或3节 + ishomegame True）或 客队向右（2节 + ishomegame False）
    loss_df['attack_dir'] = np.where(
        ((loss_df['ishomegame'] == True) & (loss_df['period'].isin([1, 3]))) |
        ((loss_df['ishomegame'] == False) & (loss_df['period'] == 2)),
        1, -1
    )

    # 坐标变换：以“向右”为正方向
    loss_df['x_plot'] = loss_df['xadjcoord'] * loss_df['attack_dir']
    loss_df['y_plot'] = loss_df['yadjcoord'] * loss_df['attack_dir']

    # 可选筛选
    if teamid_filter is not None:
        loss_df = loss_df[loss_df['teamid'] == teamid_filter]

    return loss_df[['gameid', 'teamid', 'compiledgametime', 'period', 'ishomegame',
                    'teaminpossession', 'next_teaminpossession',
                    'eventname', 'type', 'x_plot', 'y_plot']]


def visualize_possession_changes(gameid, teamid, agg_df, possession_changes):
    """
    可视化指定比赛、指定队伍的丢失球权事件位置（包含风格标签）

    参数:
        gameid: 比赛 ID
        teamid: 队伍 ID
        agg_df: 含 style 的 team-level 聚合表
        possession_changes: 来自 possession_change_events 的结果
    """
    # 获取风格标签
    team_style_row = agg_df[(agg_df['gameid'] == gameid) & (agg_df['teamid'] == teamid)]
    style = team_style_row['style'].iloc[0] if 'style' in team_style_row.columns and not team_style_row.empty else "Unknown"

    # 取出该队该场的丢失事件
    team_changes = possession_changes[(possession_changes['gameid'] == gameid) & (possession_changes['teamid'] == teamid)]
    if team_changes.empty:
        print(f"No possession losses found for team {teamid} in game {gameid}")
        return

    x_coords = team_changes['x_plot'].values
    y_coords = team_changes['y_plot'].values

    # 开始画图
    plt.figure(figsize=(12, 8))

    # 冰球场背景
    plt.plot([-100, 100, 100, -100, -100], [-42.5, -42.5, 42.5, 42.5, -42.5], 'k-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='-', linewidth=1.5, label='Center Line (x=0)')
    plt.axvline(x=25, color='b', linestyle='--', linewidth=1.2, label='Blue Line (x=±25)')
    plt.axvline(x=-25, color='b', linestyle='--', linewidth=1.2)
    circle = plt.Circle((0, 0), 15, fill=False, color='b', linewidth=1)
    plt.gca().add_patch(circle)

    # 热力图 or 散点
    if len(x_coords) > 5:
        sns.kdeplot(x=x_coords, y=y_coords, cmap="YlOrRd", fill=True, bw_adjust=0.7)
    plt.scatter(x_coords, y_coords, c='black', alpha=0.4, s=15)

    plt.title(f"Possession Loss Heatmap\nGame {gameid}, Team {teamid}, Style: {style}")
    plt.xlabel("X Coordinate (attacking right)")
    plt.ylabel("Y Coordinate")
    plt.xlim(-105, 105)
    plt.ylim(-45, 45)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_style_possession_losses(style_name, agg_df, possession_changes):
    """
    可视化指定风格下所有球队的平均丢失球权位置热力图（向右为进攻）

    参数:
        style_name: 字符串，例如 'Puck Control Play'
        agg_df: 含风格标签和 teamid/gameid 的 team-level 聚合表
        possession_changes: 来自 possession_change_events 的事件级 DataFrame
    """
    # 获取所有属于该风格的 teamid + gameid
    style_teams = agg_df[agg_df['style'] == style_name][['teamid', 'gameid']]
    if style_teams.empty:
        print(f"No teams found with style '{style_name}'")
        return

    # 合并 possession_changes 以提取所有符合该风格的事件
    merged = possession_changes.merge(style_teams, on=['teamid', 'gameid'], how='inner')
    if merged.empty:
        print(f"No possession loss events found for style '{style_name}'")
        return

    x_coords = merged['x_plot'].values
    y_coords = merged['y_plot'].values

    # 画图
    plt.figure(figsize=(12, 8))
    plt.plot([-100, 100, 100, -100, -100], [-42.5, -42.5, 42.5, 42.5, -42.5], 'k-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='-', linewidth=1.5, label='Center Line (x=0)')
    plt.axvline(x=25, color='b', linestyle='--', linewidth=1.2, label='Blue Line (x=±25)')
    plt.axvline(x=-25, color='b', linestyle='--', linewidth=1.2)
    circle = plt.Circle((0, 0), 15, fill=False, color='b', linewidth=1)
    plt.gca().add_patch(circle)

    if len(x_coords) > 10:
        sns.kdeplot(x=x_coords, y=y_coords, cmap="coolwarm", fill=True, bw_adjust=0.8)
    else:
        plt.scatter(x_coords, y_coords, c='black', alpha=0.5, s=20)

    plt.title(f"Average Possession Loss Heatmap for Style: {style_name}")
    plt.xlabel("X Coordinate (attacking right)")
    plt.ylabel("Y Coordinate")
    plt.xlim(-105, 105)
    plt.ylim(-45, 45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

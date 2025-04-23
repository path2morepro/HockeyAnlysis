from utilities import is_in_danger_zone

import numpy as np  # 记得要引入这个库，否则 np.nan 会报错

def process_hockey_data(raw_data):
    """
    综合数据处理函数：结合事件标志、射门信息、罚球信息，
    并按球队与比赛聚合统计关键指标。
    
    参数：
        raw_data: 原始冰球比赛数据（DataFrame）
        
    返回：
        处理后的 DataFrame，按 team 和 game 聚合了所有关键事件统计
    """
    
    # 拷贝一份数据，避免直接修改原始数据
    df = raw_data.copy()
    
    # ========== 1. 定义事件标志列（是否是某类事件） ==========
    df['is_pass'] = (df['eventname'] == 'pass').astype(int)  # 传球
    df['is_carry'] = (df['eventname'] == 'carry').astype(int)  # 运球
    df['is_entry'] = (df['eventname'] == 'controlledentry').astype(int)  # 控球入区
    df['is_goal'] = (df['eventname'] == 'goal').astype(int)  # 进球
    df['is_assist'] = (df['eventname'] == 'assist').astype(int)  # 助攻
    df['is_reception'] = (df['eventname'] == 'reception').astype(int)  # 接球
    df['is_puckprotect'] = df['eventname'].isin(['puckprotection', 'sopuckprotection']).astype(int)  # 护球
    df['is_block'] = (df['eventname'] == 'block').astype(int)  # 封挡
    df['is_check'] = (df['eventname'] == 'check').astype(int)  # 撞人
    df['is_entry_against'] = (df['eventname'] == 'controlledentryagainst').astype(int)  # 被控球入区
    df['is_dumpout'] = (df['eventname'] == 'dumpout').astype(int)  # 清区
    df['is_save'] = (df['eventname'] == 'save').astype(int)  # 扑救
    df['is_rebound'] = (df['eventname'] == 'rebound').astype(int)  # 门前反弹
    df['is_icing'] = (df['eventname'] == 'icing').astype(int)  # 冰球犯规（冰球未触及他人直接越过底线）
    df['is_penalty'] = (df['eventname'] == 'penalty').astype(int)  # 罚球
    df['is_penaltydrawn'] = (df['eventname'] == 'penaltydrawn').astype(int)  # 被犯规

    # ========== 2. 成功标志 ==========
    outcome_mask = (df['outcome'] == 'successful')  # 标记成功事件
    df['pass_success'] = ((df['eventname'] == 'pass') & outcome_mask).astype(int)  # 传球成功
    df['carry_success'] = ((df['eventname'] == 'carry') & outcome_mask).astype(int)  # 运球成功
    df['entry_success'] = ((df['eventname'] == 'controlledentry') & outcome_mask).astype(int)  # 入区成功

    # ========== 3. 射门相关 ==========
    df['is_shot'] = df['eventname'].isin(['shot', 'soshot']).astype(int)  # 所有类型的射门

    # 如果有 type 字段，进一步判断是否来自 slot 区域
    if 'type' in df.columns:
        df['is_slotshot'] = ((df['eventname'] == 'shot') & (df['type'] == 'slot')).astype(int)  # slot 区域射门
        df['slot_success'] = ((df['eventname'] == 'shot') & (df['type'] == 'slot') & outcome_mask).astype(int)  # slot 区域进球
    else:
        df['is_slotshot'] = 0
        df['slot_success'] = 0

    # 危险区域（球门前）射门判断
    df['is_danger_attempt'] = df.apply(lambda row: is_in_danger_zone(row['xadjcoord'], row['yadjcoord']) 
                                       if row['eventname'] == 'shot' else False, axis=1).astype(int)
    df['danger_success'] = df.apply(lambda row: is_in_danger_zone(row['xadjcoord'], row['yadjcoord']) 
                                    if row['eventname'] == 'shot' and row['outcome'] == 'successful' else False, axis=1).astype(int)

    # ========== 4. 犯规相关 ==========
    df['defence_penalty'] = ((df['eventname'] == 'penalty') & (df['xadjcoord'] < 0) & outcome_mask).astype(int)  # 防守方犯规
    df['attack_penalty'] = ((df['eventname'] == 'penalty') & (df['xadjcoord'] > 0) & outcome_mask).astype(int)  # 进攻方犯规
    df['defence_penaltydrawn'] = ((df['eventname'] == 'penaltydrawn') & (df['xadjcoord'] < 0) & outcome_mask).astype(int)  # 防守方被犯规
    df['attack_penaltydrawn'] = ((df['eventname'] == 'penaltydrawn') & (df['xadjcoord'] > 0) & outcome_mask).astype(int)  # 进攻方被犯规

    # ========== 5. 控球时间计算 ==========
    df['ownpossession'] = (df['teamid'] == df['teaminpossession']).astype(int)  # 是否为当前控球方
    df_poss = df[df['ownpossession'] == 1].copy()  # 筛选控球事件
    df_poss['time_diff'] = df_poss.groupby(['teamid', 'gameid'])['compiledgametime'].diff().fillna(0)  # 控球事件之间的时间差
    poss_time = df_poss.groupby(['teamid', 'gameid'])['time_diff'].sum().reset_index()  # 总控球时间
    poss_time.rename(columns={'time_diff': 'possession_time'}, inplace=True)

    # ========== 6. 按球队和比赛汇总 ==========
    agg_cols = [
        'is_pass', 'pass_success', 'is_carry', 'carry_success', 'is_entry', 'entry_success',
        'is_goal', 'is_assist', 'is_reception', 'is_puckprotect',
        'is_block', 'is_check', 'is_entry_against', 'is_dumpout',
        'is_save', 'is_rebound', 'is_icing', 'is_penalty', 'is_penaltydrawn',
        'is_shot', 'is_slotshot', 'slot_success', 'is_danger_attempt', 'danger_success',
        'defence_penalty', 'attack_penalty', 'defence_penaltydrawn', 'attack_penaltydrawn'
    ]
    agg_dict = {col: 'sum' for col in agg_cols}  # 所有事件都进行加总统计

    # 如果有比分差距字段，额外记录其平均值
    if 'scoredifferential' in df.columns:
        agg_dict['scoredifferential'] = 'mean'

    # 执行聚合
    agg_df = df.groupby(['teamid', 'gameid']).agg(agg_dict).reset_index()

    # 合并控球时间
    agg_df = agg_df.merge(poss_time, on=['teamid', 'gameid'], how='left')

    # ========== 7. 成功率计算 ==========
    agg_df['pass_success_rate'] = agg_df['pass_success'] / agg_df['is_pass'].replace(0, np.nan)
    agg_df['carry_success_rate'] = agg_df['carry_success'] / agg_df['is_carry'].replace(0, np.nan)
    agg_df['entry_success_rate'] = agg_df['entry_success'] / agg_df['is_entry'].replace(0, np.nan)
    agg_df['shot_success_rate'] = agg_df['is_goal'] / agg_df['is_shot'].replace(0, np.nan)
    agg_df['danger_success_rate'] = agg_df['danger_success'] / agg_df['is_danger_attempt'].replace(0, np.nan)
    agg_df['slot_success_rate'] = agg_df['slot_success'] / agg_df['is_slotshot'].replace(0, np.nan)

    # 判断比赛结果：比分差距 > 0 算胜利
    if 'scoredifferential' in agg_df.columns:
        agg_df['result'] = (agg_df['scoredifferential'] > 0).astype(int)

    # 所有 NaN 用 0 填充
    agg_df = agg_df.fillna(0)

    return agg_df

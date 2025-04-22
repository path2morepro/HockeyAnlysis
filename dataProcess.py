def shots(data):
    """
    生成以teamid为主键的球队进攻数据
    参数：
        data: 原始比赛数据
    返回：
        team_stats DataFrame
    """
    # Create computed columns
    data['is_goal'] = (data['eventname'].isin(['shot', 'soshot', 'goal'])) & (data['outcome'] == 'successful')
    data['is_shot'] = data['eventname'].isin(['shot', 'soshot'])
    data['is_danger_zone'] = data.apply(lambda row: is_in_danger_zone(row['xadjcoord'], row['yadjcoord']) 
                                    if row['eventname'] == 'shot' else False, axis=1)
    
    # 修改: 直接判断type == 'slot'而不是使用is_in_slot函数
    data['is_slot_attempt'] = (data['eventname'] == 'shot') & (data['type'] == 'slot')
    
    # 新增: slot shot且成功的情况
    data['is_slot_shot'] = (data['eventname'] == 'shot') & (data['type'] == 'slot') & (data['outcome'] == 'successful')

    # Then aggregate
    shots_stats = data.groupby('teamid').agg(
        goals=('is_goal', 'sum'),
        shots_total=('is_shot', 'sum'),
        high_danger_chances=('is_danger_zone', 'sum'),
        slot_attempts=('is_slot_attempt', 'sum'),  # 所有在slot区尝试射门的列
        slot_shots=('is_slot_shot', 'sum')  # slot shot且成功
    )
    
    # 重置索引并排序
    shots_stats = shots_stats.reset_index().set_index('teamid')
    return shots_stats
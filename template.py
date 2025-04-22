import pandas as pd

raw_data = pd.read_csv("Linhac24-25_Sportlogiq.csv")

# 开始提取特征
def create_team_offense_stats(data):
    """
    生成以teamid为主键的球队进攻数据
    参数：
        data: 原始比赛数据
    返回：
        team_stats DataFrame
    """


    """
    这里我用ai提供了一个模板
    我们可以直接用以teamid聚合
    实现我们自己的逻辑
    提取想要的特征
    我在模板里会先用 进球 中的特征做例子
    我们可以这样
    各自创建各自的函数
    但是输入输出要一样、
    确保函数的返回值是可以直接通过列拼接的
    """
    # 基本进攻指标
    team_stats = data.groupby('teamid').agg(
        # 射门相关
        shots_total=('eventname', lambda x: x.isin(['shot', 'soshot']).sum()),
        shots_successful=('outcome', lambda x: ((data['eventname'].isin(['shot', 'soshot'])) & (x == 'successful')).sum()),
        
        # 传球相关
        passes_total=('type', lambda x: x.str.contains('pass').sum()),
        passes_successful=('outcome', lambda x: ((data['type'].str.contains('pass')) & (x == 'successful')).sum()),
        
        # 控球突破
        controlled_entries=('eventname', lambda x: x.isin(['controlledentry']).sum()),
        breakthroughs=('type', lambda x: x.str.contains('carry').sum()),
        
        # 特殊战术
        faceoffs_won=('type', lambda x: x.str.contains('faceoff').sum()),
        powerplay_attempts=('manpowersituation', lambda x: x.str.contains('powerplay').sum())
    )
    
    # 计算成功率指标
    team_stats['shots_success_rate'] = team_stats['shots_successful'] / team_stats['shots_total'].replace(0, 1)
    team_stats['passes_success_rate'] = team_stats['passes_successful'] / team_stats['passes_total'].replace(0, 1)
    
    # 重置索引并排序
    team_stats = team_stats.reset_index().set_index('teamid')
    return team_stats
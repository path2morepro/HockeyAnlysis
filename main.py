import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt  
import seaborn as sns 
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 1. 读取比赛事件数据
df = pd.read_csv("Linhac24-25_Sportlogiq.csv")

# 2. 根据事件名创建新的二元变量（flag列），表示是否为某种特定事件
df['is_goal'] = (df['eventname'] == 'goal').astype(int)  # 是否为进球
df['is_shot'] = df['eventname'].isin(['shot', 'goal']).astype(int)  # 是否为射门（包含进球）
df['is_pass'] = (df['eventname'] == 'pass').astype(int)  # 是否为传球
df['is_carry'] = (df['eventname'] == 'carry').astype(int)  # 是否为控球推进
df['is_dump'] = df['eventname'].isin(['dumpin', 'dumpout']).astype(int)  # 是否为抛掷（dump）
df['is_assist'] = (df['eventname'] == 'assist').astype(int)  # 是否为助攻
df['is_slotshot'] = df['eventname'].isin(['soshot', 'sogoal']).astype(int)  # 是否为slot区的射门或进球
df['is_entry'] = (df['eventname'] == 'controlledentry').astype(int)  # 是否为带球入区
df['is_exit'] = (df['eventname'] == 'controlledexit').astype(int)  # 是否为带球出区
df['is_entryagainst'] = (df['eventname'] == 'controlledentryagainst').astype(int)  # 是否对方带球入区

# 3. 统计某些行为是否成功的标志（outcome列 == 'successful'）
df['successful_pass'] = ((df['eventname'] == 'pass') & (df['outcome'] == 'successful')).astype(int)
df['successful_carry'] = ((df['eventname'] == 'carry') & (df['outcome'] == 'successful')).astype(int)
df['successful_entry'] = ((df['eventname'] == 'controlledentry') & (df['outcome'] == 'successful')).astype(int)
df['successful_exit'] = ((df['eventname'] == 'controlledexit') & (df['outcome'] == 'successful')).astype(int)
df['successful_dump'] = ((df['eventname'].isin(['dumpin', 'dumpout'])) & (df['outcome'] == 'successful')).astype(int)

# 4. 分组汇总（按队伍、比赛和对手进行统计）
group_cols = ['teamid', 'gameid', 'opposingteamid']
agg_dict = {
    'xg_allattempts': 'sum',  # 所有射门的预期进球值
    'is_goal': 'sum', 'is_shot': 'sum', 'is_pass': 'sum', 'is_carry': 'sum',
    'is_dump': 'sum', 'is_assist': 'sum', 'is_slotshot': 'sum',
    'is_entry': 'sum', 'is_exit': 'sum', 'is_entryagainst': 'sum',
    'successful_pass': 'sum', 'successful_carry': 'sum', 'successful_entry': 'sum',
    'successful_exit': 'sum', 'successful_dump': 'sum',
    'xadjcoord': ['mean', 'std'],  # 平均x坐标和标准差（用来估计活动区域）
    'yadjcoord': ['mean', 'std'],  # 平均y坐标和标准差
    'scoredifferential': 'mean',  # 平均比分差
    'teaminpossession': lambda x: (x == x.mode()[0]).sum()  # 球队控球时间的粗略代理（最多出现的值）
}

# 聚合并重命名列名
agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]  # 多重列名展开成一维

agg_df.rename(columns={
    'xg_allattempts_sum': 'xG_total',  # 总预期进球
    'is_goal_sum': 'actual_goals',  # 实际进球数
    'is_shot_sum': 'num_shots', 'is_pass_sum': 'num_passes',
    'is_carry_sum': 'num_carries', 'is_dump_sum': 'num_dumps',
    'is_assist_sum': 'num_assists', 'is_slotshot_sum': 'num_slotshots',
    'is_entry_sum': 'num_entries', 'is_exit_sum': 'num_exits',
    'is_entryagainst_sum': 'num_entries_against',
    'successful_pass_sum': 'pass_success', 'successful_carry_sum': 'carry_success',
    'successful_entry_sum': 'entry_success', 'successful_exit_sum': 'exit_success',
    'successful_dump_sum': 'dump_success',
    'xadjcoord_mean': 'x_mean', 'xadjcoord_std': 'x_spread',
    'yadjcoord_mean': 'y_mean', 'yadjcoord_std': 'y_spread',
    'scoredifferential_mean': 'avg_score_diff',  # 平均比分差
    'teaminpossession_<lambda>': 'possession_time_proxy'  # 控球时间代理变量
}, inplace=True)

# 5. 计算成功率（成功次数除以总次数）
agg_df['pass_success_rate'] = agg_df['pass_success'] / agg_df['num_passes'].replace(0, np.nan)
agg_df['carry_success_rate'] = agg_df['carry_success'] / agg_df['num_carries'].replace(0, np.nan)
agg_df['entry_success_rate'] = agg_df['entry_success'] / agg_df['num_entries'].replace(0, np.nan)
agg_df['exit_success_rate'] = agg_df['exit_success'] / agg_df['num_exits'].replace(0, np.nan)
agg_df['dump_success_rate'] = agg_df['dump_success'] / agg_df['num_dumps'].replace(0, np.nan)

# 在这里添加了保存现存数据的代码cqx
agg_df.to_csv('basicProcess.csv', index=False, encoding='utf-8')

# 6. 选择用于建模的特征
features = [
    'xG_total', 'actual_goals', 'num_shots', 'num_passes', 'num_carries', 'num_dumps',
    'num_assists', 'num_slotshots', 'num_entries', 'num_exits', 'num_entries_against',
    'pass_success_rate', 'carry_success_rate', 'entry_success_rate',
    'exit_success_rate', 'dump_success_rate',
    'x_mean', 'y_mean', 'x_spread', 'y_spread', 'avg_score_diff', 'possession_time_proxy'
]

# 7. 特征标准化（均值为0，方差为1）
X = StandardScaler().fit_transform(agg_df[features])



'''-------------------------------------
下面这段代码有个非常不好的地方就是他已经把这个几种风格确定好了
等于说我们是在对着答案写题目
我们的思路应该是采用软聚类
就是没有提前声明k
再对数据进行聚类
这样我们的结果可能才会有和预期不一样的地方
-------------------------------------'''
# 8. 使用K均值聚类算法将球队风格分为3类
kmeans = KMeans(n_clusters=3, random_state=42)
agg_df['cluster'] = kmeans.fit_predict(X)


# 9. 给每个类别起个“打法风格”的名字（可根据聚类分析结果命名）
def label_cluster(row):
    if row['cluster'] == 0:
        return "High-Pressure Offense"  # 高压进攻型
    elif row['cluster'] == 1:
        return "Defensive Counterattack"  # 防守反击型
    elif row['cluster'] == 2:
        return "Puck Control Play"  # 控球推进型
    else:
        return "Unknown"

agg_df['style'] = agg_df.apply(label_cluster, axis=1)

# 创建胜负标签（比分差 > 0 为赢）
agg_df['result'] = (agg_df['avg_score_diff'] > 0).astype(int)

# 10. 分析不同风格之间的胜率对阵情况
merged = pd.merge(agg_df, agg_df, on='gameid', suffixes=('_team', '_opponent'))
matchups = merged[merged['teamid_team'] != merged['teamid_opponent']].copy()
matchups = matchups[['gameid', 'teamid_team', 'style_team', 'result_team', 'teamid_opponent', 'style_opponent']]

# 按照“我方风格 vs 对手风格”统计胜率
winrate_table = matchups.groupby(['style_team', 'style_opponent'])['result_team'].mean().unstack().round(2)
print("\n✅ Win Rate Matrix (Team Style vs Opponent Style):")
print(winrate_table)

# 11. 使用热力图展示风格之间的胜率矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(winrate_table, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Win Rate by Team Style vs Opponent Style")  # 标题
plt.xlabel("Opponent Style")  # 横坐标标签
plt.ylabel("Team Style")  # 纵坐标标签
plt.tight_layout()
plt.show()

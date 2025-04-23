import pandas as pd
import os
from dataProcess import *
from model import gmm, pca_select_important_features, fuzzy_cluster_pca_data
from utilities import analysis

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
first_time_run = False
pca = True
gmm_clustering = True

if first_time_run:
    # 1. Load dataset
    df = pd.read_csv("data/Linhac24-25_Sportlogiq.csv")
    df = df.sort_values(by=['gameid', 'compiledgametime'])
    agg_df = process_hockey_data(df)
    agg_df.to_csv("aggregate_stats.csv")
else:
    agg_df = pd.read_csv("data/aggregate_stats.csv")
# print(agg_df.columns)
'''
Index(['teamid', 'gameid', 'is_pass', 'pass_success', 'is_carry',
       'carry_success', 'is_entry', 'entry_success', 'is_goal', 'is_assist',
       'is_reception', 'is_puckprotect', 'is_block', 'is_check',
       'is_entry_against', 'is_dumpout', 'is_save', 'is_rebound', 'is_icing',
       'is_penalty', 'is_penaltydrawn', 'is_shot', 'is_slotshot',
       'slot_success', 'is_danger_attempt', 'danger_success',
       'defence_penalty', 'attack_penalty', 'defence_penaltydrawn',
       'attack_penaltydrawn', 'scoredifferential', 'possession_time',
       'pass_success_rate', 'carry_success_rate', 'entry_success_rate',
       'shot_success_rate', 'danger_success_rate', 'slot_success_rate',
       'result'],
      dtype='object')

'''
# 7. Features
features = [
    'is_pass', 'is_carry', 'is_entry', 'is_assist', 'is_reception',
    'is_puckprotect',
    'is_block', 'is_check', 'is_entry_against', 'is_dumpout',
    'is_save', 'is_rebound', 'is_icing',
    'pass_success_rate', 'carry_success_rate', 'entry_success_rate',
    'possession_time','shot_success_rate', 'danger_success_rate',
    'slot_success_rate','defence_penalty', 'attack_penalty',
    'defence_penaltydrawn', 'attack_penaltydrawn'
]
# 如果我们将is_goal也划分进入feature, 那肯定不会平衡的
# 因为他肯定会根据得分多的队伍进行聚类

if pca:
    # feature selection
    # PCA
    important_features, X = pca_select_important_features(data=agg_df[features], n_components=8, top_n_features=7)
    print("The most important features", important_features)
    print(X.head())
else:
    X = agg_df[features]

# 8. Soft clustering 
if gmm_clustering:
    probs = gmm(X, 3)
    agg_df[['prob_style_0', 'prob_style_1', 'prob_style_2']] = probs
    agg_df['style_soft'] = probs.argmax(axis=1)
    style_names = {0: 'High-Pressure Offense', 1: 'Defensive Counterattack', 2: 'Puck Control Play'}
    agg_df['style'] = agg_df['style_soft'].map(style_names)
else:
    # or fuzzy clustering
    labels, u, centers = fuzzy_cluster_pca_data(X, n_clusters=3)
    style_names = {0: 'S1', 1: 'S2', 2: 'S3'}
    agg_df['style'] = labels

analysis(agg_df, features)
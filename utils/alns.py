import numpy as np
import random as rd
import copy
import time
import random
from utils.dataparser import create_network_graph
from utils.metrics import common_edges_similarity_route_df_weighted
import networkx as nx

# ——— 2. 回溯破坏算子 —— 
def backtrack_destroy(history, ratio, df0, c,G,edge_index_map):
    L = len(history)-1
    if L == 0:
        return df0.copy(), None
    i = int(ratio * L)
    i = max(0, min(L, i))
    # 从快照中拷贝一次
    # 重放 0..i 的所有扰动

    for j in range(1, i+1):
        h = history[j]
        idx    = h["edge_index"]
        change = h["change_df_row"]         # dict：{col1: val1, …}
        cols   = list(change.keys())
        vals   = list(change.values())
        df0.loc[idx, cols] = vals
        u, v, k = edge_index_map[idx]
        if "include" in change:
            G.remove_edge(u, v, k)
            if G.has_edge(v,u,k):
                G.remove_edge(v,u, k)
        else:
            G[u][v][k]['my_weight'] = change["my_weight"]
            if G.has_edge(v,u,k):
                G[v][u][k]['my_weight'] = change["my_weight"]
    return i, history[i]["iteration"]

def repair_search_delete(
    idx,
    df_perturbed,
    user_model,
):
    change1 = True
    change2 = True
    if user_model["max_curb_height"] > 0.2:
        change1 = False
    if user_model["min_sidewalk_width"] > 2:
        change2 = False
    backup_row = {
        "curb_height_max": df_perturbed.at[idx, "curb_height_max"],
        "obstacle_free_width_float": df_perturbed.at[idx, "obstacle_free_width_float"],
        "include": df_perturbed.at[idx, "include"],
    }
    flag1=0
    if(change1):
        if df_perturbed.loc[idx, "curb_height_max"] <= user_model["max_curb_height"]:
            backup_row["curb_height_max"] = 0.2
            backup_row["include"] = 0
        else:
            if(change2):
                if df_perturbed.loc[idx, "obstacle_free_width_float"] >= user_model["min_sidewalk_width"]:
                    backup_row["obstacle_free_width_float"] = 0.6
                    backup_row["include"] = 0
                    # print(2)
    weight_delta = df_perturbed.loc[idx, "my_weight"]
    score = weight_delta
    return score, idx, backup_row



def repair_search_change_weight(
    idx,
    df_perturbed,
    user_model,
    maxx_weight
):
    weight_delta=df_perturbed.at[idx, "my_weight"]
    backup_row = {
        "path_type": df_perturbed.at[idx, "path_type"],
        "my_weight": df_perturbed.at[idx, "my_weight"],
    }
    if df_perturbed.loc[idx, "path_type"] != user_model["walk_bike_preference"]:
            backup_row["my_weight"]=df_perturbed.at[idx, "length"]
            if df_perturbed.loc[idx, "crossing"] == "Yes":
                backup_row["my_weight"] *= user_model["crossing_weight_factor"]
            backup_row["my_weight"]*=user_model["walk_bike_preference_weight_factor"]
            backup_row["my_weight"]/=maxx_weight
            backup_row["path_type"]=user_model["walk_bike_preference"]
            weight_delta-=backup_row["my_weight"]
    score = weight_delta 
    return score, idx, backup_row
def pick_max_score(valid_sorted):
    """取分数最高的（第 0 个）"""
    return valid_sorted[0]

def pick_min_score(valid_sorted):
    """取分数最低的（最后一个）"""
    return valid_sorted[-1]

def pick_median_score(valid_sorted):
    """取中位数位置的那条"""
    mid = len(valid_sorted) // 2
    return valid_sorted[mid]

def pick_random(valid_sorted):
    """随机挑一条"""
    return random.choice(valid_sorted)
# # ——— 3. 局部扰动序列（你自己实现） —— 
# def apply_your_perturb_sequence(df_base):
#     df_new = df_base.copy()
#     # TODO: 在这里对 df_new 做一次或多次“边属性扰动”或删边操作
#     #    例如：随机挑一条 fact-only 边把 include=0，然后 handle_weight/update 图
#     return df_new
# def repair_delete_edge(df_base):
#     """
#     随机挑一条 fact-only 边，把 include=0（删边）
#     """
#     df = df_base.copy()
#     fact_only = list(set(df_fact_path["edge_idx"]) - set(df_foil_all["edge_idx"]))
#     if fact_only:
#         idx = rd.choice(fact_only)
#         df.loc[idx, "include"] = 0
#     return df
# def repair_adjust_weight(df_base):
#     """
#     随机挑一条 fact-only 边，把它的 curb_height_max 和 width 恢复到阈值（放行）
#     """
#     df = df_base.copy()
#     fact_only = list(set(df_fact_path["edge_idx"]) - set(df_foil_all["edge_idx"]))
#     if fact_only:
#         idx = rd.choice(fact_only)
#         df.loc[idx, "curb_height_max"] = user_model["max_curb_height"]
#         df.loc[idx, "obstacle_free_width_float"] = user_model["min_sidewalk_width"]
#         df.loc[idx, "include"] = 1
#     return df

# # ——— 4. 主程序 —— 
# if __name__ == '__main__':
#     # 初始解
#     df_current = initial_df_perturbed.copy()
#     last_err, last_w, df_fact_path = evaluate_solution(df_current)
#     best_df = df_current.copy()
#     best_err, best_w = last_err, last_w

#     # 历史轨迹
#     perturb_history = [df_current.copy()]

#     T = T_init

#     for outer in range(iterMax):
#         while T > 10:
#             # — a) 破坏：回溯到某次扰动
#             base_df, back_idx = backtrack_destroy(perturb_history, T)
#             # 截断历史
#             perturb_history = perturb_history[:back_idx+1]
#             destroyUseTime[0] += 1
#             while(没有达到相似度要求):
# 				# 2) 选择修复算子（轮盘赌）
# 				r = rd.uniform(0, wRepair[0] + wRepair[1])
# 				if r < wRepair[0]:
# 					df_new = repair_delete_edge(base_df, T)
# 					op_idx = 0
# 				else:
# 					df_new = repair_adjust_weight(base_df, T)
# 					op_idx = 1
# 				repairUseTime[op_idx] += 1

#             # — c) 评估新解
#             err_new, w_new, new_fact = evaluate_solution(df_new)
#             if err_new is None:
#                 accept = False
#             else:
#                 Δerr  = last_err - err_new
#                 Δw    = last_w   - w_new
#                 λ     = 0.5
#                 score = λ*Δerr + (1-λ)*Δw

#                 if score >= 0:
#                     accept = True
#                     destroyScore[0] += 1.5
#                     repairScore[0]  += 1.5
#                 else:
#                     p = np.exp(score / T)
#                     if rd.random() < p:
#                         accept = True
#                         destroyScore[0] += 0.8
#                         repairScore[0]  += 0.8
#                     else:
#                         accept = False
#                         destroyScore[0] += 0.6
#                         repairScore[0]  += 0.6

#             # — d) 接受新解
#             if accept:
#                 df_current   = df_new
#                 last_err     = err_new
#                 last_w       = w_new
#                 df_fact_path = new_fact
#                 perturb_history.append(df_current.copy())
#                 # 更新全局最优
#                 if last_err < best_err or (last_err==best_err and last_w<best_w):
#                     best_df, best_err, best_w = df_current.copy(), last_err, last_w

#             # — e) 更新算子权重
#             wDestroy[0] = (1-b)*(destroyScore[0]/destroyUseTime[0]) + b*wDestroy[0]
#             wRepair[0]  = (1-b)*(repairScore[0]/ repairUseTime[0]) + b*wRepair[0]

#             # — f) 降温
#             T *= alpha

#         # 重置温度，进行下一轮外循环
#         T = T_init

#     # 输出最优结果
#     print("Best error:", best_err, " Best weight:", best_w)
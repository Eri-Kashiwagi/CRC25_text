import os
import argparse
import json
# Standard library and path imports
import sys
import os
import time
import json
sys.path.append(".")
# Third-party library imports
import numpy as np
import pandas as pd
import shapely.ops as so
import shapely.geometry as sg
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import networkx as nx
import momepy
import random as rd
import random
# Local or project-specific imports
from utils.helper import get_modified_edges_df
from router import Router
from utils.graph_op import graphOperator
from utils.dataparser import  create_network_graph, handle_weight, handle_weight_with_recovery
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
import multiprocessing as mp
from copy import deepcopy 
from shapely import wkt
from utils.mthread import generate_neighbor_p, parallel_generate_neighbor
import multiprocessing as mp
from utils.parallel_utils import eval_edge_top
from utils.alns import backtrack_destroy
from utils.alns import pick_max_score
from utils.alns import pick_min_score
from utils.alns import pick_median_score
from utils.alns import pick_random
from utils.alns import repair_search_delete
from utils.alns import repair_search_change_weight
from utils.dataparser import store_op_list, load_op_list
from utils.graph_op import pertub_with_op_list
from utils.dataparser import convert
from shapely import to_wkt
import json
class LS():
    def __init__(self, args):
        self.args = args
    
    def reset(self):
        self.heuristic = self.args['heuristic']
        self.heuristic_f = self.args['heuristic_f']
        self.attrs_variable_names = self.args['attrs_variable_names']
        self.jobs = self.args['jobs']
        if self.jobs > 1:
            self.pool = mp.Pool(processes=self.jobs)
        else:
            self.pool = None
        

        self.df, self.path_foil, self.df_path_foil, self.gdf_coords_loaded = read_data(self.args)
        self.user_model = self.args["user_model"]
        self.meta_map = self.args["meta_map"]
        df_copy = deepcopy(self.df)
        df_copy,self.maxx_weight = handle_weight(df_copy, self.user_model)
        _, self.G = create_network_graph(df_copy)
        self.df=df_copy

        self.router_h= Router(heuristic=self.heuristic, CRS=self.meta_map["CRS"], CRS_map=self.meta_map["CRS_map"])
        self.graph_operator = graphOperator()
        self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, self.gdf_coords = self.router_h.set_o_d_coords(self.G, self.gdf_coords_loaded)

        self.path_fact, self.G_path_fact, self.df_path_fact = self.router_h.get_route(self.G, self.origin_node, self.dest_node, self.heuristic_f)
    
    def generate_neighbor(self, df):
        (df_perturbed_i, G_perturbed_i),(df_path_i, G_path_i), op_list_perturbed = generate_neighbor_p(df, self.router_h, self.graph_operator, self.origin_node, self.dest_node, self.args, self.user_model)
        if df_perturbed_i is None:
            return (None, None, 0), (None, None, 0), op_list_perturbed
        
        sub_op_list = get_virtual_op_list(self.df, df_perturbed_i, self.attrs_variable_names)
        graph_error = len([op for op in sub_op_list if op[3] == "success"])

        route_error = 1-common_edges_similarity_route_df_weighted(df_path_i, self.df_path_foil, self.attrs_variable_names)
        return (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), (op_list_perturbed, sub_op_list)

    
    def generate_population(self, df, pop_num):
        pop = []
        if self.jobs > 1:
            jobs = [self.pool.apply_async(parallel_generate_neighbor, (df, self.router_h, self.graph_operator, self.origin_node, self.dest_node, self.df_path_foil, self.args, self.user_model, )) for _ in range(pop_num)]
            for idx, j in enumerate(jobs):
                try:
                    (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), op_lists = j.get()
                except Exception as e:
                    print(e)
                    (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), op_lists = (0, 0, 0), (0, 0, 0), []

                pop.append(((df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), op_lists))
        else:
            for _ in range(pop_num):
                pop.append((self.generate_neighbor(df)))
        return pop

    def get_perturbed_edges(self, df_perturbed):
        modified_edges_df = get_modified_edges_df(self.df, df_perturbed, self.attrs_variable_names)
        return modified_edges_df


def read_data(args):

    basic_network_path = args['basic_network_path']
    foil_json_path = args['foil_json_path']
    df_path_foil_path = args['df_path_foil_path']
    gdf_coords_path = args['gdf_coords_path']

    df = gpd.read_file(basic_network_path)
    with open(foil_json_path, 'r') as f:
        path_foil = json.load(f)

    df_path_foil = gpd.read_file(df_path_foil_path)
    gdf_coords_loaded = pd.read_csv(gdf_coords_path, sep=';')

    gdf_coords_loaded['geometry'] = gdf_coords_loaded['geometry'].apply(wkt.loads)
    gdf_coords_loaded = gpd.GeoDataFrame(gdf_coords_loaded, geometry='geometry')

    return df, path_foil, df_path_foil, gdf_coords_loaded

def get_results(args):
    # TODO: Implement this function with your own algorithm
    start_time = time.time()

    # —— 构造各类文件路径 —— #
    basic_network_path = args.basic_network_path
    foil_json_path     = args.foil_json_path
    df_path_foil_path  = args.df_path_foil_path
    gdf_coords_path    = args.gdf_coords_path
    meta_data_path     = getattr(args, 'meta_data_path', None)
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    # Profile settings
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]

    attrs_variable_names = user_model["attrs_variable_names"]
    route_error_delta = user_model["route_error_threshold"]
    # Demo route

    n_perturbation = 50
    operator_p = [0.15, 0.15, 0.15, 0.15, 0.4]
    args1 = {
        'basic_network_path': basic_network_path,
        'foil_json_path': foil_json_path,
        'df_path_foil_path': df_path_foil_path,
        'gdf_coords_path': gdf_coords_path,
        'heuristic': 'dijkstra',
        'heuristic_f': 'my_weight',
        'jobs': 1,
        'attrs_variable_names': attrs_variable_names,
        "n_perturbation": n_perturbation,
        "operator_p": operator_p,
        "user_model": user_model,
        "meta_map": meta_map
    }

    # 1.1 实例化 LS 并 reset，内部已经调用 handle_weight 和 create_network_graph
    ls = LS(args1)
    ls.reset()

    # 1.2 从 LS 拿到图 G（有向、多重图），以及原始的 DF 和 foil 路径 DF
    G = ls.G  # 由 create_network_graph(handle_weight(df)) 生成，图上每条边上的属性已包含 my_weight

    iteration = 0
    change1=True
    change2=True
    change3=True
    change4=True
    iteration = 0
    ls.df["edge_idx"] = ls.df.index
    df_perturbed = ls.df.copy()
    df_perturbed1=ls.df.copy()

    df_full_idx = df_perturbed.set_index("geometry", drop=False)
    df_foil_idx = ls.df_path_foil.set_index("geometry")
    full_only_cols = df_full_idx.columns.difference(df_foil_idx.columns)
    maxx_weight=df_perturbed['my_weight'].abs().max()
    # 只把这些 “独有” 列拼过来
    df_foil_all = df_foil_idx.join(
        df_full_idx[full_only_cols],
        how="left"
    ).reset_index(drop=True)
    _, df_G1 = create_network_graph(df_perturbed)
    for idx, foil_row in ls.df_path_foil.iterrows():
        # idx 是 ls.df_path_foil 里这一行的索引
        foil_geom = foil_row["geometry"]

        # 在完整网络表 ls.df 里匹配相同 geometry，拿到对应行号列表
        orig_matches = ls.df.index[ls.df["geometry"] == foil_geom].tolist()

        if len(orig_matches) == 0 or len(orig_matches) > 1:
            continue

        orig_idx = orig_matches[0]

        # 把找到的 orig_idx 写回到 ls.df_path_foil 的 edge_idx 列
        ls.df_path_foil.loc[idx, "edge_idx"] = orig_idx

        # 以下如果想还原 df_perturbed 中这条边的 include/属性，也可以用 orig_idx 去操作 df_perturbed
        if df_perturbed.loc[orig_idx, "include"] == 0:
            if df_perturbed.loc[orig_idx, "curb_height_max"] > user_model["max_curb_height"]:
                df_perturbed.loc[orig_idx, "curb_height_max"] = user_model["max_curb_height"]
                iteration += 1

            if df_perturbed.loc[orig_idx, "obstacle_free_width_float"] < user_model["min_sidewalk_width"]:
                df_perturbed.loc[orig_idx, "obstacle_free_width_float"] = user_model["min_sidewalk_width"]
                iteration += 1

            df_perturbed.loc[orig_idx, "include"] = 1
    df_path_foil = ls.df_path_foil 
    ls.df=df_perturbed.copy()
    G_con_dir, G_sel_con_dir = create_network_graph(df_perturbed)
    ls.G = G_sel_con_dir
    df_G=ls.G
    edge_index_map = {}
    for u, v, key, data in ls.G.edges(keys=True, data=True):
        idx = data.get("edge_idx")
        if idx is not None:
            edge_index_map[idx] = (u, v, key)
    fact_path, G_fact_path, df_fact_path = ls.router_h.get_route(
        ls.G, ls.origin_node, ls.dest_node, ls.heuristic_f
    )
    ttt_G=ls.G.copy()
    sim_old = common_edges_similarity_route_df_weighted(df_fact_path, ls.df_path_foil, attrs_variable_names)
    df_fact_path1=df_fact_path.copy()
    route_error_old = 1.0 - sim_old
    if route_error_old <= route_error_delta:
        # …如果你想要在这里输出最终结果，可以补上
        exit(0)
    # 进入到正式的迭代循环
    last_route_error=route_error_old
    last_weight=df_fact_path["my_weight"].sum()
    cnt=0
    gen_log  = []   # 每次对某条边扰动后的记录
    best_log = []   # 每轮选出的最佳扰动记录
    new_route_error=None
    all_iteration_status=[]
    all_iteration_status.append({
    "iteration":     0,
    "edge_index":    None,     # 没有被删改的那一行
    "change_df_row": {},       # 空 dict，表示没有列要更新
    "last_error":          route_error_old,
    "fact_path":     df_fact_path
    })
    while True:
        foil_edge_idxs = set(df_foil_all["edge_idx"])
        fact_edge_idxs = set(df_fact_path["edge_idx"])
        # 2. 计算三类边
        common_edges      = foil_edge_idxs & fact_edge_idxs        # 交集：既在 foil 也在 fact 中的边
        foil_only_edges   = foil_edge_idxs - fact_edge_idxs        # 只在 foil 中出现的边
        fact_only_edges   = fact_edge_idxs - foil_edge_idxs        # 只在 fact 中出现的边
        df_common = df_perturbed.loc[list(common_edges)]
        df_foil_only = df_perturbed.loc[list(foil_only_edges)]
        df_fact_only = df_perturbed.loc[list(fact_only_edges)]
        # ---------- 贪心删边 ----------
        best_score = -float("inf")
        best_idx = None
        best_fact_path = None
        best_df = None
        if(new_route_error!=None):
            last_route_error=new_route_error
        inputs = [
            (
                idx,                  # 1. idx
                df_perturbed,         # 2. df_perturbed
                df_fact_path,         # 3. df_fact_path
                df_path_foil,      # 4. df_path_foil
                user_model,           # 5. user_model
                last_route_error,     # 6. last_route_error
                attrs_variable_names, # 7. attrs_variable_names
                ls.origin_node,       # 8. origin_node
                ls.dest_node,         # 9. dest_node
                ls.router_h.get_route, # 10. router_fun
                df_G,
                edge_index_map,
                route_error_delta,
            )
            for idx in fact_only_edges
        ]

        if args1["jobs"] > 1:
            with mp.Pool(processes=args1["jobs"]) as pool:
                results = pool.starmap(eval_edge_top, inputs)
        else:
            results = [eval_edge_top(*inp) for inp in inputs]

        valid = [r for r in results if r is not None]
        if not valid:
            break

        best_score, best_idx, change_df_row, best_fact_path, best_err = max(
            valid, key=lambda x: x[0]
        )
        if best_idx is None:
            break

        # 真正生效
        df_perturbed.loc[best_idx, change_df_row.keys()] = list(change_df_row.values())
        df_fact_path = best_fact_path
        new_route_error=best_err
        u, v, k = edge_index_map[best_idx]
        df_G.remove_edge(u, v, k)
        if df_G.has_edge(v,u,k):
            flag1=1
            df_G.remove_edge(v,u, k)
        cnt += 1
        ttt=time.time()
        if best_err <= route_error_delta:
            break
        all_iteration_status.append({
            "iteration":cnt,
            "change_df_row": change_df_row,
            "edge_index":best_idx,
            "last_error":best_err,
            "fact_path":best_fact_path
        })
        if(time.time()-start_time>270):
            break
    cnt2=0
    last_route_error=route_error_old
    last_weight=df_fact_path1["my_weight"].sum()
    new_route_error=None
    ungood=0
    while cnt>=cnt2:
        tt=time.time()
        foil_edge_idxs = set(df_foil_all["edge_idx"])
        fact_edge_idxs = set(df_fact_path1["edge_idx"])
        # 2. 计算三类边
        common_edges      = foil_edge_idxs & fact_edge_idxs        # 交集：既在 foil 也在 fact 中的边
        foil_only_edges   = foil_edge_idxs - fact_edge_idxs        # 只在 foil 中出现的边
        fact_only_edges   = fact_edge_idxs - foil_edge_idxs        # 只在 fact 中出现的边
        # 3. 打印一下规模，确认无误
        df_common = df_perturbed1.loc[list(common_edges)]
        df_foil_only = df_perturbed1.loc[list(foil_only_edges)]
        df_fact_only = df_perturbed1.loc[list(fact_only_edges)]
        # ---------- 贪心删边 ----------
        best_score = -float("inf")
        best_idx = None
        best_fact_path = None
        best_df = None
        if(new_route_error!=None):
            last_route_error=new_route_error
        inputs = [
            (
                idx,                  # 1. idx
                df_perturbed1,         # 2. df_perturbed
                df_fact_path1,         # 3. df_fact_path
                df_path_foil,      # 4. df_path_foil
                user_model,           # 5. user_model
                last_route_error,     # 6. last_route_error
                attrs_variable_names, # 7. attrs_variable_names
                ls.origin_node,       # 8. origin_node
                ls.dest_node,         # 9. dest_node
                ls.router_h.get_route, # 10. router_fun
                df_G1,
                edge_index_map,
                route_error_delta,
            )
            for idx in fact_only_edges
        ]

        if args1["jobs"] > 1:
            with mp.Pool(processes=args1["jobs"]) as pool:
                results = pool.starmap(eval_edge_top, inputs)
        else:
            results = [eval_edge_top(*inp) for inp in inputs]

        valid = [r for r in results if r is not None]
        if not valid:
            break

        best_score, best_idx, change_df_row, best_fact_path, best_err = max(
            valid, key=lambda x: x[0]
        )
        if best_idx is None:
            break
        # 真正生效
        df_perturbed1.loc[best_idx, change_df_row.keys()] = list(change_df_row.values())
        df_fact_path1 = best_fact_path
        new_route_error=best_err
        u, v, k = edge_index_map[best_idx]
        df_G1.remove_edge(u, v, k)
        if df_G1.has_edge(v,u,k):
            flag1=1
            df_G1.remove_edge(v,u, k)
        cnt2 += 1
        ttt=time.time()
        if best_err <= route_error_delta:
            ungood=1
            break
        if(time.time()-start_time>270):
            break
    min_iteration=cnt+iteration
    route_error_min=-999999
    ddd=0
    if(cnt!=1):
        new_route_error=None

        #---------------------------------------
        #进行ALNS破除局部
        T_init = 130      # 初始温度
        alpha  = 0.996      # 降温速度
        T_min  = 10        # 最小温度
        b      = 0.5       # 算子权重更新系数
        c      = 0.05      # backtrack 权重函数中的调节参数

        num_destroy = 4
        num_repair  = 8

        # 破坏算子
        wDestroy      = [1.0]*num_destroy
        imp_sum_destroy   = [0.0] * num_destroy
        imp_count_destroy = [0]   * num_destroy
        subScore_destroy  = [0.0] * num_destroy
        destroy_operators=[0,0.25,0.5,0.75]
        destroyUseTime=[0.0] * num_destroy

        wRepair       = [1.0]*num_repair
        repairUseTime = [0]*num_repair
        repairScore   = [1.0]*num_repair
        Repair_son_sub=[0.2,0.2,0.2,0.4]
        #八个修复算子
        num_subops  = 8                 # 子算子总数
        wSub        = [1.0] * num_subops  # 权重（轮盘赌用）
        subUseTime  = [0]   * num_subops  # 使用次数
        subScore    = [1.0] * num_subops  # 积分（>0 保证第一次除数不为 0）
        last_subUseTime=subUseTime.copy()
        b_sub       = 0.7                # 子算子权重平滑系数

        fullCalls   = [0] * num_subops   # accept==2 的使用次数
        annealCalls = [0] * num_subops   # accept==1 的使用次数
        rejCalls    = [0] * num_subops   # accept==0 的使用次数
        # 主循环
        T = T_init
        ddd=0
        base_score=cnt+iteration
        best_iteration=[]
        for rec in all_iteration_status:
            best_iteration.append({
                "iteration":     rec["iteration"],
                "change_df_row": rec["change_df_row"].copy(),  # dict.copy()
                "edge_index":    rec["edge_index"],
                "last_error":    rec["last_error"],
                "fact_path":     rec["fact_path"].copy()       # DataFrame.copy()
            })
        weight_history = []
        weight_history1=[]
        count_history  = []
        while T > T_min:
            ddd+=1
            df0= ls.df.copy()
            G=ttt_G.copy()
            destroy_idx = random.choices(range(len(destroy_operators)), weights=wDestroy, k=1)[0]
            ratio = destroy_operators[destroy_idx]
            chosen_idx,now_iteration = backtrack_destroy(
                history = all_iteration_status,
                ratio=ratio,
                df0     = df0,
                c       = c,
                G=G,
                edge_index_map=edge_index_map,
            )
            slice_hist=[]
            for h in all_iteration_status[:chosen_idx+1]:
                slice_hist.append({
                    "iteration":      h["iteration"],
                    "change_df_row":  h["change_df_row"].copy(),    # dict浅拷贝
                    "edge_index":     h["edge_index"],
                    "last_error":     h["last_error"],
                    "fact_path":      h["fact_path"].copy(),        # DataFrame.copy()
                })
            # _,G = create_network_graph(df0)
            v_op_list = get_virtual_op_list(ls.df, df0, args1["attrs_variable_names"])
            count=0
            df_fact_path=all_iteration_status[chosen_idx]["fact_path"]
            last_route_error=all_iteration_status[chosen_idx]["last_error"]
            while(True):
                new_record=0
                foil_edge_idxs = set(df_foil_all["edge_idx"])
                fact_edge_idxs = set(df_fact_path["edge_idx"])
                # 2. 计算三类边
                common_edges      = foil_edge_idxs & fact_edge_idxs        # 交集：既在 foil 也在 fact 中的边
                foil_only_edges   = foil_edge_idxs - fact_edge_idxs        # 只在 foil 中出现的边
                fact_only_edges   = fact_edge_idxs - foil_edge_idxs        # 只在 fact 中出现的边
                # 3. 打印一下规模，确认无误
                df_common = df0.loc[list(common_edges)]
                df_foil_only = df0.loc[list(foil_only_edges)]
                df_fact_only = df0.loc[list(fact_only_edges)]
                best_score = -float("inf")
                best_idx = None
                best_fact_path = None
                best_df = None
                if(new_route_error!=None):
                    last_route_error=new_route_error
                op_idx = random.choices(range(num_subops), weights=wSub, k=1)[0]
                if op_idx > 3:
                    df_foil_only_temp = df0.loc[list(foil_only_edges)]
                    df_foil_only_temp = df_foil_only_temp[
                        df_foil_only_temp['path_type'] != user_model['walk_bike_preference']
                    ]
                    if len(df_foil_only_temp) == 0:
                        # 强制切换到删边算子 (0-3)
                        op_idx = random.choices(range(4), weights=wSub[:4], k=1)[0]
                if op_idx <= 3:
                    inputs = [
                    (
                        idx,                  # 1. idx
                        df0,         # 2. df_perturbed
                        user_model,           # 5. user_model
                    )
                    for idx in fact_only_edges
                    ]
                    results = [repair_search_delete(*inp) for inp in inputs]
                    valid = [r for r in results if r is not None]
                    valid_sorted = sorted(valid, key=lambda x: x[0], reverse=True)
                    pickers = [pick_max_score, pick_min_score, pick_median_score, pick_random]
                    chosen_op = pickers[op_idx]
                    # 3) 用它从 valid 里挑出 (score, idx, change, fact_path, err)
                    best_score, best_idx, change_df_row= chosen_op(valid)
                    df0.loc[best_idx, change_df_row.keys()] = list(change_df_row.values())
                    u, v, k = edge_index_map[best_idx]
                    G.remove_edge(u, v, k)
                    if G.has_edge(v,u,k):
                        G.remove_edge(v,u, k)
                    _, _, best_fact_path = ls.router_h.get_route(
                        G, ls.origin_node, ls.dest_node, ls.heuristic_f
                    )
                    sim_old = common_edges_similarity_route_df_weighted(best_fact_path, ls.df_path_foil, attrs_variable_names)
                    best_err=1-sim_old
                    # 4) 应用改动到 df_curr，得到 df_new
                    # 真正生效
                    df_fact_path = best_fact_path
                    new_route_error=best_err
                    subUseTime[op_idx]+=1
                    count += 1

                else:
                    foil_edge_useful_idxs = set(df_foil_only_temp["edge_idx"])
                    inputs = [
                    (
                        idx,                  # 1. idx
                        df0,         # 2. df_perturbed
                        user_model,           # 5. user_model
                        ls.maxx_weight,
                    )
                    for idx in foil_edge_useful_idxs
                    ]
                    results = [repair_search_change_weight(*inp) for inp in inputs]
                    valid = [r for r in results if r is not None]
                    valid_sorted = sorted(valid, key=lambda x: x[0], reverse=True)
                    pickers = [pick_max_score, pick_min_score, pick_median_score, pick_random]
                    chosen_op = pickers[op_idx-4]

                    # 3) 用它从 valid 里挑出 (score, idx, change, fact_path, err)
                    best_score, best_idx, change_df_row = chosen_op(valid)
                    df0.loc[best_idx, change_df_row.keys()] = list(change_df_row.values())
                    u, v, k = edge_index_map[best_idx]
                    G[u][v][k]["my_weight"]=change_df_row["my_weight"]
                    if G.has_edge(v,u,k):
                        G[u][v][k]["my_weight"]=change_df_row["my_weight"]
                    _, _, best_fact_path = ls.router_h.get_route(
                        G, ls.origin_node, ls.dest_node, ls.heuristic_f
                    )
                    sim_old = common_edges_similarity_route_df_weighted(best_fact_path, ls.df_path_foil, attrs_variable_names)
                    best_err=1-sim_old
                    # 4) 应用改动到 df_curr，得到 df_new
                    # 真正生效
                    df_fact_path = best_fact_path
                    new_route_error=best_err
                    count += 1
                    subUseTime[op_idx]+=1
                if best_err <= route_error_delta:
                    break
                slice_hist.append({
                    "iteration":count+now_iteration,
                    "change_df_row": change_df_row,
                    "edge_index":best_idx,
                    "last_error":best_err,
                    "fact_path":best_fact_path
                })
                # all_iteration_status[:] = slice_hist

            # — c) 评估
            now_score=count+now_iteration+iteration
            if base_score>now_score:
                all_iteration_status[:] = slice_hist
                base_score=now_score
                accept = 2
                subScore_destroy[destroy_idx]+=1
                if(len(best_iteration)>len(all_iteration_status)):
                    min_iteration=count+now_iteration+iteration
                    best_iteration.clear()
                    for rec in all_iteration_status:
                        best_iteration.append({
                            "iteration":     rec["iteration"],
                            "change_df_row": rec["change_df_row"].copy(),  # dict.copy()
                            "edge_index":    rec["edge_index"],
                            "last_error":    rec["last_error"],
                            "fact_path":     rec["fact_path"].copy()       # DataFrame.copy()
                        })
                    sim_old = common_edges_similarity_route_df_weighted(df_fact_path, ls.df_path_foil, attrs_variable_names)
                    route_error_min = 1.0 - sim_old
                    df_perturbed=df0.copy()
                    subScore_destroy[destroy_idx]+=5
                    new_record=2
                if(len(best_iteration)==len(all_iteration_status)):
                    best_iteration.clear()
                    for rec in all_iteration_status:
                        best_iteration.append({
                            "iteration":     rec["iteration"],
                            "change_df_row": rec["change_df_row"].copy(),  # dict.copy()
                            "edge_index":    rec["edge_index"],
                            "last_error":    rec["last_error"],
                            "fact_path":     rec["fact_path"].copy()       # DataFrame.copy()
                        })
                    subScore_destroy[destroy_idx]+=3
                    new_record=1

            else:
                p = np.exp(15*(base_score-now_score-1) / T)
                if rd.random() < p:
                    accept = 1
                    subScore_destroy[destroy_idx]+=0.1
                    all_iteration_status[:] = slice_hist
                    base_score+=(now_score-base_score)/2
                else:
                    accept = 0
                    subScore_destroy[destroy_idx]-=0.1
                    
            gamma = 0.08   # 惩罚系数
            eps   = 1e-6
            tau0_d       = 1.0
            min_tau_d    = 0.1
            mu0_d        = 0.3
            mu_min_d     = 0.05
            MAX_ITERS    = 600
            # 平滑更新：用 (1–b)*平均得分 + b*旧权重
            # destroyScore 用 subScore_destroy 累积打分
            # 1) 进度与温度
            progress = min(1.0, ddd / float(MAX_ITERS))
            tau_d    = max(min_tau_d, tau0_d * (1 - progress))

            # 2) 清洗 subScore_destroy，防止 NaN/负值
            eps_score = 1e-6
            subScore_destroy = [
                eps_score if (not np.isfinite(s) or s < eps_score) else s
                for s in subScore_destroy
            ]

            # 3) 数值稳定版 Softmax
            scores     = np.array(subScore_destroy, dtype=float)
            max_score  = np.max(scores)
            shifted    = (scores - max_score) / tau_d
            exp_scores = np.exp(shifted)               # 全部 <= 1
            sum_exp    = exp_scores.sum() or 1.0
            w_norm_d   = (exp_scores / sum_exp).tolist()

            # 4) 混合探索：保留 μ 均匀概率
            mu     = mu0_d * (1 - progress) + mu_min_d * progress
            u_prob = 1.0 / num_destroy
            wDestroy = [(1 - mu) * w + mu * u_prob for w in w_norm_d]

            # 5) 归一化一次，消除浮点误差
            s = sum(wDestroy) or 1.0
            wDestroy = [wd / s for wd in wDestroy]

            weight_history1.append(wDestroy.copy())

            # —— 3) 按“全接受／退火接受／拒绝”比率更新 subScore —— 
            for i in range(num_subops):
                calls = subUseTime[i]
                if accept == 2:
                    fullCalls[i]   += calls
                elif accept == 1:
                    annealCalls[i] += calls
                else:
                    rejCalls[i]    += calls
                p_full   = fullCalls[i] 
                p_anneal = annealCalls[i] 
                p_reject = rejCalls[i]  
                # reward = 1*p_full + 0.5*p_anneal – γ*p_reject
                reward = p_full + 0.07 * p_anneal - gamma * p_reject
                if(new_record==2):
                    reward+=subUseTime[i]*17
                if(new_record==1):
                    reward+=subUseTime[i]*5
                subScore[i] = max(
                    eps,
                    (1 - b_sub) * subScore[i]
                    + b_sub     * reward
                )
            
            # —— 4) 归一化得到新权重 wSub —— 
            total = sum(subScore)
            wSub   = [s/total for s in subScore]
            total = sum(subScore)
            if total > 0:
                wSub = [s / total for s in subScore]
            else:
                wSub = [1.0/num_subops] * num_subops

            progress = ddd / MAX_ITERS
            mu0    = 0.3   # 初始探索强度（30% 均匀探索）
            mu_min = 0.05  # 最终保留探索比例（5% 均匀探索）
            mu     = mu0 * (1 - progress) + mu_min * progress

            # U 是均匀分布
            u_prob = 1.0 / num_subops
            # 混合
            wSub = [(1 - mu) * w + mu * u_prob for w in wSub]
            # —— 5) 清空本轮调用统计 —— 
            subUseTime = [0] * num_subops
            weight_history.append(wSub.copy())
            count_history.append(count + iteration+now_iteration)
            # —— 6) 退火降温 —— 
            T *= alpha
            if(time.time()-start_time>270):
                break

    end_time = time.time()
    duration = end_time - start_time
    if(ungood==1):
        if(min_iteration>cnt2):
            df_perturbed=df_perturbed1
    ls.reset()
    v_op_list = get_virtual_op_list(ls.df, df_perturbed, args1["attrs_variable_names"])
    available_op = [(op[0], (convert(op[1][0]), to_wkt(op[1][1], rounding_precision=-1, trim=False)), convert(op[2]), op[3]) for op in v_op_list if op[3] == "success"]
    map_df = df_perturbed
    op_list = available_op
    
    return map_df, op_list



def store_results(output_path, map_df, op_list):
    
    map_df_path = os.path.join(output_path, "map_df.gpkg")
    op_list_path = os.path.join(output_path, "op_list.json")

    map_df.to_file(map_df_path, driver='GPKG')
    with open(op_list_path, 'w') as f:
        json.dump(op_list, f)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_data_path", type=str, required=True)
    parser.add_argument("--basic_network_path", type=str, required=True)
    parser.add_argument("--foil_json_path", type=str, required=True)
    parser.add_argument("--df_path_foil_path", type=str, required=True)
    parser.add_argument("--gdf_coords_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    map_df, op_list = get_results(args)
    store_results(args.output_path, map_df, op_list)
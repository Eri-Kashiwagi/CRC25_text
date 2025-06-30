from utils.dataparser import create_network_graph
from utils.metrics import common_edges_similarity_route_df_weighted

def eval_edge_top(
    idx,
    df_perturbed,
    df_fact_path,
    df_path_foil,
    user_model,
    last_route_error,
    attrs_variable_names,
    origin_node,
    dest_node,
    router_fun,       # <-- 新增一个参数，用来接 ls.router_h.get_route
    df_G,
    edge_index_map,
    route_error_delta,
):
    """
    完全按照原 eval_edge 逻辑：复制 df，删边扰动，重建图，重新路由，打印调试，计算 score，记录 gen_log
    """
    change1 = True
    change2 = True
    if user_model["max_curb_height"] > 0.2:
        change1 = False
    if user_model["min_sidewalk_width"] > 2:
        change2 = False
    u, v, k = edge_index_map[idx]
    data_backup = df_G[u][v][k].copy()
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
            df_G.remove_edge(u, v, k)
            if not data_backup.get("oneway", True):
                flag1=1
                df_G.remove_edge(v,u, k)
        else:
            if(change2):
                if df_perturbed.loc[idx, "obstacle_free_width_float"] >= user_model["min_sidewalk_width"]:
                    backup_row["obstacle_free_width_float"] = 0.6
                    backup_row["include"] = 0
                    df_G.remove_edge(u, v, k)
                    if not data_backup.get("oneway", True):
                        flag1=1
                        df_G.remove_edge(v,u, k)
    
    try:
        # _,G_tmp = create_network_graph(df_tmp)
        fact_path_new, _, df_fact_path_new = router_fun(
            df_G,
            origin_node,
            dest_node,
            'my_weight'
        )
    except Exception as e:
        # 恢复图
        df_G.add_edge(u, v, key=k, **data_backup)
        if flag1:
            df_G.add_edge(v, u, key=k, **data_backup)
        return None
    df_fact_path_new_set = set(df_fact_path_new["edge_idx"])
    df_fact_path_set = set(df_fact_path["edge_idx"])
    dist=df_fact_path_new["my_weight"].sum()
    weight_delta = dist-df_fact_path["my_weight"].sum()
    sim_new = common_edges_similarity_route_df_weighted(df_fact_path_new, df_path_foil, attrs_variable_names)
    route_error_new = 1.0 - sim_new
    last_route_error_delta=last_route_error-route_error_new
    if(last_route_error_delta<0):
        last_route_error_delta=0
    score = weight_delta + last_route_error_delta  # 调参，怎么组合自己玩
    if(route_error_new<route_error_delta):
        score+=10000
    df_G.add_edge(u, v, key=k, **data_backup)
    if(flag1==1):
        df_G.add_edge(v, u, key=k, **data_backup)
    return score, idx, backup_row, df_fact_path_new, route_error_new

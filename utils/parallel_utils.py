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
):
    """
    完全按照原 eval_edge 逻辑：复制 df，删边扰动，重建图，重新路由，打印调试，计算 score，记录 gen_log
    """
    # # 1) 打印 idx
    # print(idx)

    # # 2) 复制 DataFrame
    # df_tmp = df_perturbed.copy()

    # # 3) 原始 change1/change2 标志
    # change1 = True
    # change2 = True
    # if user_model["max_curb_height"] > 0.2:
    #     change1 = False
    # if user_model["min_sidewalk_width"] > 2:
    #     change2 = False

    # # 4) 扰动逻辑（按 include=0 演示）
    # if change1:
    #     if df_tmp.loc[idx, "curb_height_max"] <= user_model["max_curb_height"]:
    #         df_tmp.loc[idx, "curb_height_max"] = user_model["max_curb_height"]
    #         df_tmp.loc[idx, "include"] = 0
    #         print(1)
    #     else:
    #         if change2:
    #             if df_tmp.loc[idx, "obstacle_free_width_float"] >= user_model["min_sidewalk_width"]:
    #                 df_tmp.loc[idx, "obstacle_free_width_float"] = user_model["min_sidewalk_width"]
    #                 df_tmp.loc[idx, "include"] = 0
    #                 print(2)
    #             else:
    #                 print("无法扰动，error")
    #         else:
    #             print("无法扰动，error")
    # else:
    #     print("无法扰动，error")

    # # 5) 重建图 & 路由
    # try:
    #     _, G_tmp = create_network_graph(df_tmp)
    #     fact_path_new, _, df_fact_path_new = router_fun(
    #         G_tmp,
    #         origin_node,
    #         dest_node,
    #         'dijkstra'
    #     )
    # except Exception as e:
    #     print(f"删边 {idx} 后不可达/异常，跳过: {e}")
    #     return None

    # # 6) 打印 fact-only 与 公共边
    # df_new_set = set(df_fact_path_new["edge_idx"])
    # df_old_set = set(df_fact_path["edge_idx"])
    # print("fact-only-new edges:", df_new_set - df_old_set)
    # print("fact-only-old edges:", df_old_set - df_new_set)
    # print("common edges:", df_new_set & df_old_set)

    # # 7) 计算 Δweight
    # dist = df_fact_path_new["my_weight"].sum()
    # weight_delta = dist - df_fact_path["my_weight"].sum()

    # # 8) 计算 route_error
    # sim_new = common_edges_similarity_route_df_weighted(
    #     df_fact_path_new,
    #     df_path_foil,
    #     attrs_variable_names
    # )
    # route_error_new = 1.0 - sim_new

    # # 9) 计算 score
    # last_route_error_delta = last_route_error - route_error_new
    # score = weight_delta + last_route_error_delta
    df_tmp = df_perturbed.copy()
    change1 = True
    change2 = True
    if user_model["max_curb_height"] > 0.2:
        change1 = False
    if user_model["min_sidewalk_width"] > 2:
        change2 = False
    if(change1):
        if df_tmp.loc[idx, "curb_height_max"] <= user_model["max_curb_height"]:
            df_tmp.loc[idx, "curb_height_max"] = 0.2
            df_tmp.loc[idx, "include"] = 0
            print(1)
        else:
            if(change2):
                if df_tmp.loc[idx, "obstacle_free_width_float"] >= user_model["min_sidewalk_width"]:
                    df_tmp.loc[idx, "obstacle_free_width_float"] = 0.6
                    df_tmp.loc[idx, "include"] = 0
                    print(2)
                else:
                    print("无法扰动，error")
            else:
                print("无法扰动，error")
    else:
        print("无法扰动，error")
    # 你的扰动逻辑可以更细，比如阈值，这里按include=0简单演示
    # 建图
    
    try:
        _,G_tmp = create_network_graph(df_tmp)
        fact_path_new, _, df_fact_path_new = router_fun(
            G_tmp,
            origin_node,
            dest_node,
            'my_weight'
        )
    except Exception as e:
        print(f"删边 {idx} 后不可达/异常，跳过: {e}")
    df_fact_path_new_set = set(df_fact_path_new["edge_idx"])
    df_fact_path_set = set(df_fact_path["edge_idx"])
    print("fact-only-new edges:", df_fact_path_new_set - df_fact_path_set)
    print("fact-only-old edges:", df_fact_path_set - df_fact_path_new_set)
    print("common edges:", df_fact_path_new_set & df_fact_path_set)
    dist=df_fact_path_new["my_weight"].sum()
    weight_delta = dist-df_fact_path["my_weight"].sum()
    sim_new = common_edges_similarity_route_df_weighted(df_fact_path_new, df_path_foil, attrs_variable_names)
    route_error_new = 1.0 - sim_new
    last_route_error_delta=last_route_error-route_error_new
    score = weight_delta + last_route_error_delta  # 调参，怎么组合自己玩

    return score, idx, df_tmp, df_fact_path_new, route_error_new

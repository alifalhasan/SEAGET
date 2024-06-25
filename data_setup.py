import os

import pandas as pd
import numpy as np
import networkx as nx

from tqdm.auto import tqdm
from pathlib import Path

# Hyperparameters
ALPHA = 0.67
BETA = 0.67


def build_POI_graph(df):
    users = list(set(df["user_id"].to_list()))

    node_dict = {}
    edge_dict = {}

    # Filter out the checkin month
    df["checkin_month"] = pd.to_datetime(df["UTC_time"]).dt.month

    for user_id in tqdm(users):
        user_df = df[df["user_id"] == user_id]

        previous_node = 0
        previous_traj_id = 0

        for i, row in user_df.iterrows():
            # Adding nodes
            node = row["POI_id"]
            if node not in node_dict:
                node_dict[node] = {
                    "POI_id": node,
                    "checkin_cnt_past": 0,
                    "checkin_cnt_recent": 0,
                    "user_cnt_past": set(),
                    "user_cnt_recent": set(),
                    "poi_catid": row["POI_catid"],
                    "poi_catid_code": row["POI_catid_code"],
                    "poi_catname": row["POI_catname"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                }
            if row["checkin_month"] >= 9:
                node_dict[node]["checkin_cnt_recent"] += 1
                node_dict[node]["user_cnt_recent"].add(user_id)
            else:
                node_dict[node]["checkin_cnt_past"] += 1
                node_dict[node]["user_cnt_past"].add(user_id)

            # Adding edges
            traj_id = row["trajectory_id"]
            if (previous_node == 0) or (previous_traj_id != traj_id):
                previous_node = node
                previous_traj_id = traj_id
                continue

            # Add edges
            if (previous_node, node) not in edge_dict:
                edge_dict[(previous_node, node)] = {
                    "x": previous_node,
                    "y": node,
                    "checkin_cnt_recent": 0,
                    "checkin_cnt_past": 0,
                    "user_cnt_past": set(),
                    "user_cnt_recent": set(),
                }
            if row["checkin_month"] >= 9:
                edge_dict[(previous_node, node)]["checkin_cnt_recent"] += 1
                edge_dict[(previous_node, node)]["user_cnt_recent"].add(user_id)
            else:
                edge_dict[(previous_node, node)]["checkin_cnt_past"] += 1
                edge_dict[(previous_node, node)]["user_cnt_past"].add(user_id)
            previous_traj_id = traj_id
            previous_node = node

    node_list = list(node_dict.values())
    edge_list = list(edge_dict.values())

    # Creating the graph
    G = nx.DiGraph()

    # Add nodes
    for node in node_list:
        if node["POI_id"] not in G.nodes():
            G.add_node(
                node["POI_id"],
                weight=BETA
                * (
                    ALPHA * len(node["user_cnt_recent"])
                    + (1.0 - ALPHA) * node["checkin_cnt_recent"]
                )
                + (1.0 - BETA)
                * (
                    ALPHA * len(node["user_cnt_past"])
                    + (1.0 - ALPHA) * node["checkin_cnt_past"]
                ),
                poi_catid=node["poi_catid"],
                poi_catid_code=node["poi_catid_code"],
                poi_catname=node["poi_catname"],
                latitude=node["latitude"],
                longitude=node["longitude"],
            )

    # Add edges
    for edge in edge_list:
        if G.has_edge(edge["x"], edge["y"]):
            print("jhamela ache")
        else:
            G.add_edge(
                edge["x"],
                edge["y"],
                weight=BETA
                * (
                    ALPHA * len(edge["user_cnt_recent"])
                    + (1.0 - ALPHA) * edge["checkin_cnt_recent"]
                )
                + (1.0 - BETA)
                * (
                    ALPHA * len(edge["user_cnt_past"])
                    + (1.0 - ALPHA) * edge["checkin_cnt_past"]
                ),
            )

    return G


def save_graph_to_csv(G):
    # Saves the graph in two different files: adj_mat and node_list
    # adj_mat file: edge from (i -> j) with weight, rows and columns are ordered according to node_list file
    # node_list file: same order with adj_mat file

    data_path = Path("data")
    processed_path = data_path / "processed"

    if processed_path.is_dir() == False:
        processed_path.mkdir(parents=True, exist_ok=True)

    # Save adj_mat
    nodelist = G.nodes()
    adj_mat = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(
        os.path.join(processed_path, "adj_mat.csv"), adj_mat.todense(), delimiter=","
    )

    # Save node_list
    nodelist = list(G.nodes.data())
    with open(processed_path / "node_features.csv", "w") as f:
        print(
            "node_name/poi_id,weight,poi_catid,poi_catid_code,poi_catname,latitude,longitude",
            file=f,
        )
        for node in nodelist:
            node_name = node[0]
            weight = node[1]["weight"]
            poi_catid = node[1]["poi_catid"]
            poi_catid_code = node[1]["poi_catid_code"]
            poi_catname = node[1]["poi_catname"]
            latitude = node[1]["latitude"]
            longitude = node[1]["longitude"]
            print(
                f"{node_name},{weight},{poi_catid},{poi_catid_code},{poi_catname},{latitude},{longitude}",
                file=f,
            )


if __name__ == "__main__":
    # Read raw data
    train_df = pd.read_csv(r"data/processed/NYC_train.csv")

    # Build POI graph
    G = build_POI_graph(train_df)

    # Save graph to disk
    save_graph_to_csv(G)

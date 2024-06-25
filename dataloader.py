import numpy as np
import pandas as pd


def load_graph_adj_mat(path):
    """adj_mat.shape: (num_node * num_node)"""
    adj_mat = np.loadtxt(path, delimiter=",")
    return adj_mat


def load_graph_node_features(
    path,
    feature1="weight",
    feature2="poi_catid_code",
    feature3="latitude",
    feature4="longitude",
):
    """node_features.shape: (num_node, 4), Currently considering four features: weight, category, latitude, logitude"""
    df = pd.read_csv(path, encoding="latin-1")
    feature_df = df[[feature1, feature2, feature3, feature4]]
    node_features = feature_df.to_numpy()
    return node_features

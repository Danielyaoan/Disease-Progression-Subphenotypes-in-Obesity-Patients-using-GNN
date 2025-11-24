import numpy as np
import pandas as pd
import pickle
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans 
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import MinMaxScaler

###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################

import numpy as np
from itertools import combinations
import bisect
import concurrent.futures
import joblib
from tslearn.clustering import silhouette_score
from sklearn.metrics import davies_bouldin_score
from tslearn.metrics import cdist_dtw
import argparse
import pandas as pd
from torch_geometric.data import Data
import numpy as np

import os
import pickle
import argparse
import json
from types import SimpleNamespace
# Set the random seed for reproducibility
#np.random.seed(42)

def execute(cfg, pfe):
    cluster = cfg.n_clusters
    print('cluster', cluster)
            
    #loading orginal data
    path_dataset = ".../graph_output_bmi_1&2_sub/output_node_features.pickle"
    # loading files
    file_dataset = open(path_dataset, 'rb')
    raw_df = pickle.load(file_dataset)
    raw_df.reset_index(inplace=True)
    temp_df  = raw_df[['PATID', 'ENCID']]

    #Read the indices from the CSV
    #read_indices = pd.read_csv("../Utils/Sampling_index/train_indices.csv")["index"].tolist()
    root_path = ".../Output_bmi_1&2_avg_min_sub/"
    dataset = 'whole_dataset'
    file_name = "features_embedding.npy"
    model = "Magnet"


    # List of input files and their corresponding names
    files = {
        "GCN": root_path + 'GCN' + "/"+ dataset +"/"+ file_name,
        "GAT": root_path + 'GAT' + "/"+ dataset +"/"+ file_name,
        "GraphSAGE": root_path + 'GraphSAGE' + "/"+ dataset +"/"+ file_name,
        "Magnet": root_path + 'Magnet' + "/"+ dataset +"/"+ file_name
    }

    path = files[model]
    print(model, path)

    # Load the .npy file
    data = np.load(path)
    # Print the shape of the loaded data
    print(f"Shape of {model} data embedings: {data.shape}")
    # Use these indices to get the sampled rows from the data
    embedding_feature = data
    print(f"Shape of {model} Train set data embedings: {embedding_feature.shape}")

    # Assuming the new features are named 'Feature1' and 'Feature2'
    embedding_feature_length = embedding_feature.shape[1]
    feature_df = pd.DataFrame(embedding_feature, columns=['Feature'+str(i) for i in range(embedding_feature_length)])
    # Concatenate temp_df with the feature embedding DataFrame
    temp_df = pd.concat([temp_df, feature_df], axis=1)
    print(temp_df)

    # preprocess data 
    # Preprocess train set
    temp_df_train = temp_df#.iloc[read_indices, : ]


    temp_df_train_sorted_df = temp_df_train.sort_values(by=['PATID', 'ENCID'])
    # Group by 'PATID' and aggregate each group's features into a numpy array
    grouped_features = temp_df_train_sorted_df.groupby('PATID').apply(lambda group: group[['Feature' + str(i) for i in range(embedding_feature_length)]].values)
    sequences = [group for group in grouped_features]
    X = to_time_series_dataset(np.array(sequences, dtype="object"))
    print('Train set shape after preprocess:', X.shape)

    np.random.seed(42)
    resampled_indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0]), replace=False)
    X = X[resampled_indices, :]
    # Print the shape of the sampled data
    print(f"Shape of sampled {model} data: {X.shape}")


    # 定义参数网格
    param_grid = {
        "kmeans__n_clusters": [cluster],
        "metric": [ "dtw", "softdtw"],
        # "max_iter": [50, 100, 200, 400],
        "n_init": [5, 10, 15, 20],
        # "tol":[1e-6, 1e-5, 1e-7, 1e-8]
    }

    scores = []
    best_score = 0
    best_params = None
    best_model = None
    row = []
    # 遍历参数组合
    for k in param_grid["kmeans__n_clusters"]:
        for n_init in param_grid["n_init"]:
            for metric in param_grid["metric"]:
                # for max_iter in param_grid["max_iter"]:
                    # for tol in param_grid["tol"]:
                        print(k)
                        # ts_kmeans = TimeSeriesKMeans(n_clusters=k, n_init=n_init, metric=metric, max_iter=max_iter, tol=tol, random_state=42)
                        ts_kmeans = TimeSeriesKMeans(n_clusters=k,  metric=metric, n_init=n_init, random_state=42)
                        labels = ts_kmeans.fit_predict(X)

                        score = silhouette_score(X, labels)
                        scores.append(score)
                        print(f"silhouette_score score for k = {k}: {score}")
                        # print(f'params n_init {n_init}; metric {metric}; max_iter {max_iter}; tol {tol}')
                        print(f'params metric {metric}; n_init {n_init};')
                        row.append([k, metric, n_init, score])

                        # 选择最佳参数（inertia 越小越好）
                        if score > best_score:
                            best_score = score
                            # best_params = {"n_clusters": k, "metric": metric, "max_iter": max_iter,  "n_init": n_init, "tol": tol}
                            best_model = ts_kmeans
    # joblib.dump(best_model, f"ts_kmeans_model_{model}.pkl")
    print(f"best params: {best_params}")
    print(f"best score: {best_score}")
    temp = pd.DataFrame(row, columns=['k', 'metric', 'n_init', 'score'])
    temp.to_csv(f'./score_results/search_magnet_k={k}_ninit.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)
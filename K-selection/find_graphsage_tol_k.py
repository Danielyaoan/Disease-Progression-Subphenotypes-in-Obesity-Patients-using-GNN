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
from sklearn.metrics import silhouette_score as skl_silhouette_score
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
# 定义参数网格
# Set the random seed for reproducibility
#np.random.seed(42)

################################################################################################################################################
import psutil, os, gc
from joblib import Parallel, delayed
from tqdm import tqdm
import math
import time

def print_memory(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # in GB
    print(f"[MEMORY] {note}: {mem:.2f} GB")
    gc.collect()  # force garbage collection to see real usage
    
def build_dtw_matrix_tiled(
    X,
    block=1024,           # tile size (rows/cols per tile)
    n_jobs=20,
    dtype=np.float32,
    out_path=None         # e.g., "/dev/shm/dtw_float32.dat" (RAM disk) or None for in-RAM ndarray
):
    """
    Compute full pairwise DTW distances using 2D tiles, in parallel.
    Returns a numpy.memmap (if out_path given) or a numpy.ndarray otherwise.
    """
    n = X.shape[0]

    # Choose storage: memmap (recommended for very large n) or in-RAM array if you truly have enough memory
    if out_path:
        D = np.memmap(out_path, mode="w+", dtype=dtype, shape=(n, n))
    else:
        D = np.empty((n, n), dtype=dtype)

    ntiles = math.ceil(n / block)
    tiles = []
    for bi in range(ntiles):
        i0 = bi * block
        i1 = min((bi + 1) * block, n)
        for bj in range(bi, ntiles):  # only upper triangle (bj >= bi)
            j0 = bj * block
            j1 = min((bj + 1) * block, n)
            tiles.append((i0, i1, j0, j1))

    def compute_tile(i0, i1, j0, j1):
        A = X[i0:i1]
        B = X[j0:j1]
        # inner single-thread DTW; outer parallel handles concurrency
        Dij = cdist_dtw(A, B, n_jobs=1).astype(dtype, copy=False)
        D[i0:i1, j0:j1] = Dij
        if (j0 != i0) or (j1 != i1):  # mirror to lower triangle when off-diagonal
            D[j0:j1, i0:i1] = Dij.T
        return (i0, i1, j0, j1)

    # Parallel over tiles
    Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(compute_tile)(i0, i1, j0, j1) for (i0, i1, j0, j1) in tqdm(tiles, desc="DTW blocks")
    )

    # Zero diagonal (numerically it should already be ~0)
    for bi in range(ntiles):
        i0 = bi * block
        i1 = min((bi + 1) * block, n)
        np.fill_diagonal(D[i0:i1, i0:i1], 0.0)

    return D


def cdist_dtw_with_progress(X, block=256, n_jobs=20, dtype=np.float32):
    """
    Compute full DTW distance matrix with a progress bar, block by block.
    Returns a dense numpy array (n x n).
    """
    n = len(X)
    D = np.empty((n, n), dtype=dtype)

    for i0 in tqdm(range(0, n, block), desc="DTW blocks"):
        i1 = min(i0+block, n)
        D[i0:i1, :] = cdist_dtw(X[i0:i1], X, n_jobs=n_jobs).astype(dtype, copy=False)

    # Symmetrize and fix diagonal
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
################################################################################################################################################
    

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
    model = "GraphSAGE"

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
    
    del data

    # Assuming the new features are named 'Feature1' and 'Feature2'
    embedding_feature_length = embedding_feature.shape[1]
    feature_df = pd.DataFrame(embedding_feature, columns=['Feature'+str(i) for i in range(embedding_feature_length)])
    # Concatenate temp_df with the feature embedding DataFrame
    temp_df = pd.concat([temp_df, feature_df], axis=1)
    print(temp_df)

    # preprocess data 
    # Preprocess train set
    temp_df_train = temp_df#.iloc[read_indices, : ]
    
    del embedding_feature, feature_df, temp_df


    temp_df_train_sorted_df = temp_df_train.sort_values(by=['PATID', 'ENCID'])
    # Group by 'PATID' and aggregate each group's features into a numpy array
    grouped_features = temp_df_train_sorted_df.groupby('PATID').apply(lambda group: group[['Feature' + str(i) for i in range(embedding_feature_length)]].values)
    sequences = [group for group in grouped_features]
    X = to_time_series_dataset(np.array(sequences, dtype="object"))
    print('Train set shape after preprocess:', X.shape)
    
    del temp_df_train, temp_df_train_sorted_df, grouped_features, sequences

    np.random.seed(42)
    resampled_indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0]), replace=False)
    X = X[resampled_indices, :]
    # Print the shape of the sampled data
    print(f"Shape of sampled {model} data: {X.shape}")
    
    print_memory("Before KMeans")


    param_grid = {
        "kmeans__n_clusters": [cluster],
        "metric": [ "dtw"],#, "softdtw"],
        # "max_iter": [50, 100, 200, 400],
        # "n_init": [5, 10, 15, 20],
        "tol":[1e-6]#, 1e-7, 1e-8]
    }

    scores = []
    best_score = 0
    best_params = None
    best_model = None
    row1 = []
    row2 = []
    # 遍历参数组合
    for k in param_grid["kmeans__n_clusters"]:
        # for n_init in param_grid["n_init"]:
            for metric in param_grid["metric"]:
                # for max_iter in param_grid["max_iter"]:
                    for tol in param_grid["tol"]:
                        print(k)
                        # ts_kmeans = TimeSeriesKMeans(n_clusters=k, n_init=n_init, metric=metric, max_iter=max_iter, tol=tol, random_state=42)
                        t0 = time.time()
                        ts_kmeans = TimeSeriesKMeans(n_clusters=k,  metric=metric, tol=tol, random_state=42)
                        labels = ts_kmeans.fit_predict(X)
                        t1 = time.time()
                        print(f"KMeans Elapsed time = {(t1 - t0)/60:.2f} minutes")
                        
                        ################################################################################################################################################
                        print_memory("After KMeans")
                        
                        #X = X.astype(np.float32, copy=False)
                        X = np.asarray(X, dtype=np.float32, order="C")
                        
                        print_memory("Before Distance 1 Computed")
                        
                        
                        N = len(X)
                        rng1 = np.random.default_rng(42)
                        idx1 = rng1.choice(N, size=10_000, replace=False)
                        rng2 = np.random.default_rng(0)
                        idx2 = rng2.choice(N, size=10_000, replace=False)

                        X_sub1 = X[idx1]
                        labels_sub1 = labels[idx1]
                        
                        X_sub2 = X[idx2]
                        labels_sub2 = labels[idx2]
                        
                        t0 = time.time()
                        #D1 = cdist_dtw(X_sub1, X_sub1, n_jobs=20)     # shape (10k, 10k), ~400 MB float32
                        D1 = cdist_dtw_with_progress(X_sub1, block=512, n_jobs=20)  # you can adjust block
                        t1 = time.time()
                        print(f"DTW Distance Elapsed time 1 = {(t1 - t0)/60:.2f} minutes")
                        
                        print_memory("After Distance 1 Computed")
                        
                        t0 = time.time()
                        #D2 = cdist_dtw(X_sub2, X_sub2, n_jobs=20)     # shape (10k, 10k), ~400 MB float32
                        D2 = cdist_dtw_with_progress(X_sub2, block=512, n_jobs=20)  # you can adjust block
                        t1 = time.time()
                        print(f"DTW Distance Elapsed time 2 = {(t1 - t0)/60:.2f} minutes")
                        
                        #D = cdist_dtw(X, X, n_jobs=-1)          # float64 by default
                        #D = build_dtw_matrix_tiled(X, block=2048, n_jobs=20, dtype=np.float32, out_path=None)
                        
                        print_memory("After Distance 2 Computed")
                        
                        t0 = time.time()
                        score1 = skl_silhouette_score(D1, labels_sub1, metric="precomputed")
                        t1 = time.time()
                        print(f"silhouette_score Elapsed time = {(t1 - t0)/60:.2f} minutes") 
                        
                        t0 = time.time()
                        score2 = skl_silhouette_score(D2, labels_sub2, metric="precomputed")
                        t1 = time.time()
                        print(f"silhouette_score 2 Elapsed time = {(t1 - t0)/60:.2f} minutes") 
                     ################################################################################################################################################

                        #score = silhouette_score(X, labels)
                        #scores.append(score1)
                        print(f"silhouette_score 1 score for k = {k}: {score1}")
                        print(f"silhouette_score 2 score for k = {k}: {score2}")
                        # print(f'params n_init {n_init}; metric {metric}; max_iter {max_iter}; tol {tol}')
                        print(f'params metric {metric}; tol {tol};')
                        row1.append([k, metric, tol, score1])
                        row2.append([k, metric, tol, score2])


                        # 选择最佳参数（inertia 越小越好）
                        # if score > best_score:
                        #     best_score = score
                        #     # best_params = {"n_clusters": k, "metric": metric, "max_iter": max_iter,  "n_init": n_init, "tol": tol}
                        #     best_model = ts_kmeans
                        # joblib.dump(best_model, f"ts_kmeans_model_{model}.pkl")
                        #print(f"best params: {best_params}")
                        #print(f"best score: {best_score}")
                        temp = pd.DataFrame(row1, columns=['k', 'metric', ' tol', 'score'])
                        temp.to_csv(f'./score_results/search_graphsage_k={k}_tol1.csv', index=False)
                        temp2 = pd.DataFrame(row2, columns=['k', 'metric', ' tol', 'score'])
                        temp2.to_csv(f'./score_results/search_graphsage_k={k}_tol2.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)
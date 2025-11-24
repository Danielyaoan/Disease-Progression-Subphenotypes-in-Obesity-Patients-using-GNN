# We assume that PyTorch is already installed
import torch
torchversion = torch.__version__
import pandas as pd
from torch_geometric.data import Data
import numpy as np
import os
import pickle
from Utils import io as Utils
import argparse
import json
from types import SimpleNamespace
from Models import GCN, GAT, GraphSAGE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from tslearn.utils import to_time_series_dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def execute(cfg, pfe):
    # loading all parameter
    root_dir = cfg.root_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    processed_dir = cfg.dataset_dir
    output_root_dir = cfg.output_root_dir
    output_features_dir = cfg.output_features_dir
    features_embedding_file_name = cfg.features_embedding_file_name
    output_cluster_dir = cfg.output_cluster_dir
    cluster_filename = cfg.cluster_filename

    path_dataset = os.path.join(root_dir, processed_dir, cfg.dataset_filename)
    path_adj_matrix = os.path.join(root_dir, processed_dir,cfg.adj_matrix_filename)

    #cluster settings
    n_clusters = cfg.n_clusters
    distance_matrix = cfg.distance_matrix
    linkage = cfg.linkage


    # loading data
    _, _, _, _, raw_df = Utils.load_data(path_dataset, path_adj_matrix, 1)
    temp_df  = raw_df[['PATID', 'ENCID']]

    #Read the indices from the CSV
    #read_indices = pd.read_csv("")["index"].tolist()

    # loading learned embedding features
    embedding_feature = Utils.load_numpy(output_root_dir, output_features_dir, model_name, model_id, features_embedding_file_name)
    print('embedding_feature shapes : ',embedding_feature.shape)
    embedding_feature_length = embedding_feature.shape[1]

    # Assuming the new features are named 'Feature1' and 'Feature2'
    feature_df = pd.DataFrame(embedding_feature, columns=['Feature'+str(i) for i in range(embedding_feature_length)])
    # Concatenate temp_df with the feature embedding DataFrame
    temp_df = pd.concat([temp_df, feature_df], axis=1)
    print(temp_df)

    # Preprocess train set
    temp_df_train = temp_df#.iloc[read_indices, : ]

    
    temp_df_train_sorted_df = temp_df_train.sort_values(by=['PATID', 'ENCID'])
    # Group by 'PATID' and aggregate each group's features into a numpy array
    grouped_features = temp_df_train_sorted_df.groupby('PATID').apply(lambda group: group[['Feature' + str(i) for i in range(embedding_feature_length)]].values)
    sequences = [group for group in grouped_features]
    X = to_time_series_dataset(np.array(sequences))
    print('Train set shape:', X.shape)

    # Train the clustering
    Cluster = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw",random_state=42).fit(X)

    # Preprocess the output set
    temp_sorted_df = temp_df.sort_values(by=['PATID', 'ENCID'])
    # Group by 'PATID' and aggregate each group's features into a numpy array
    # Group by 'PATID' and collect both feature values and their indices
    grouped_features_with_indices = temp_sorted_df.groupby('PATID').apply(
        lambda group: (
            group[['Feature' + str(i) for i in range(embedding_feature_length)]].values,  # Feature values
            group.index.values  # Indices
        )
    )
    sequences = [group[0] for group in grouped_features_with_indices]
    sequence_indices = [temp_sorted_df['PATID'][group[1]].unique()[0] for group in grouped_features_with_indices]
    Xtest = to_time_series_dataset(np.array(sequences))
    print('Output set shape:', Xtest.shape)

    # get the predict 
    labels = Cluster.predict(Xtest)
   
    #assgin label to original data
    temp_df['cluster_info'] = np.NaN 
    for i, patid in enumerate(sequence_indices):
        temp_df.loc[temp_df['PATID'] == patid, 'cluster_info'] = labels[i]
    print(temp_df)
    print(temp_df['cluster_info'].value_counts())
    temp_df = temp_df[['PATID','ENCID','cluster_info']]
    # save result to file
    Utils.save_dataframe(temp_df, output_root_dir, output_cluster_dir, model_name, model_id, cluster_filename)
    






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)

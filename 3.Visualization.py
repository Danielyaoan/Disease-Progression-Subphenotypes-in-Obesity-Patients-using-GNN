# We assume that PyTorch is already installed
import torch
torchversion = torch.__version__
import pandas as pd
from torch_geometric.data import Data
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from Utils import io as Utils
import argparse
import json
from types import SimpleNamespace
from mpl_toolkits import mplot3d
from Models import GCN, GAT, GraphSAGE
import matplotlib.lines as mlines

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def execute(cfg, pfe):
    # loading all parameter
    root_dir = cfg.root_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    output_root_dir = cfg.output_root_dir
    output_features_dir = cfg.output_features_dir
    features_embedding_file_name = cfg.features_embedding_file_name
    output_cluster_dir = cfg.output_cluster_dir
    cluster_filename = cfg.cluster_filename
    output_fig_dir = cfg.output_fig_dir
    

    method_name = cfg.method_name
    n_components = cfg.n_components
    random_state = cfg.random_state
    node_size = cfg.node_size


    # loading learned embedding features
    embedding_feature = Utils.load_numpy(output_root_dir, output_features_dir, model_name, model_id, features_embedding_file_name)
    
    #Read the indices from the CSV
    read_indices = pd.read_csv(".../Graphmodel/GraphModeling/Utils/Sampling_index/Sub-phenotype/test_indices.csv")["index"].tolist()
    # Select a subset of the features for fitting based on the read indices
    embedding_feature = embedding_feature[read_indices]

    #loading clustering info
    df_cluster = Utils.load_dataframe(output_root_dir, output_cluster_dir, model_name, model_id, cluster_filename)
    df_cluster['cluster_info'] = df_cluster['cluster_info'].astype(int)
    cluster = df_cluster['cluster_info'].values
    cluster = cluster[read_indices]
    
    
    n_cluster = len(df_cluster['cluster_info'].unique())
    """
    path_dataset = ""
    file_dataset = open(path_dataset, 'rb')
    raw_df = pickle.load(file_dataset)
    raw_df['current_status'] = raw_df['current_status'].astype(int)
    cluster = raw_df['current_status'].values
    n_cluster = len(raw_df['current_status'].unique())
    """
    for n_component in n_components:
    
        output_fig_filename = cfg.method_name+str(n_component)+"D"+cfg.output_fig_filename

        #loading method and get decompsition result
        decom_method = Utils.get_decomposition_method(method_name, n_component, random_state)
        decom_results = decom_method.fit_transform(embedding_feature)

        x = decom_results[:, 0]
        y = decom_results[:, 1]

        if n_component >= 3:
            z = decom_results[:, 2]
            plt.figure(figsize=(12, 10))
            axes = plt.axes(projection='3d')
            
            # Custom color mapping
            colors = ['#673ca8', '#C34f77', '#96af77', '#6491c7', '#F38c16', '#FA40B6', '#A84972', '#AC04BF']
            colors = colors[:n_cluster]
            cluster_colors = [colors[c]  for c in cluster]

            scatter = axes.scatter3D(x, y, z, c=cluster_colors, s=node_size)
            # Adding a legend
            legend_labels = [f'Subphenotype {i+1}' for i in range(len(colors))]
            #legend_labels = ['Pre_MCI', 'MCI', 'AD']
            legend_handles = [mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=10) for i in range(len(legend_labels))]
            # Adding a legend with larger font size
            legend = axes.legend(handles=legend_handles, labels=legend_labels, title='Cluster', fontsize='xx-large', title_fontsize=20)
            
            # Adjusting the legend position
            #legend.set_bbox_to_anchor((0.2, 0.8))

            # Hiding the axis numbers
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            axes.set_zticklabels([])
            #save fig
            plt.tight_layout()
            Utils.save_fig(plt, output_root_dir, output_fig_dir, model_name, model_id, output_fig_filename)
            
        elif n_component ==2:
            plt.figure(figsize=(12, 10))
            
            colors = ['#673ca8', '#C34f77', '#96af77', '#6491c7', '#F38c16', '#FA40B6', '#A84972', '#AC04BF']
            colors = colors[:n_cluster]
            cluster_colors = [colors[c]  for c in cluster]
            
            scatter = plt.scatter(x, y, c=cluster_colors, s=node_size)
             # Adding a legend
            legend_labels = [f'Subphenotype {i+1}' for i in range(len(colors))]
            #legend_labels = ['Pre_MCI', 'MCI', 'AD']
            legend_handles = [mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=10) for i in range(len(legend_labels))]
            legend = plt.legend(handles=legend_handles, labels=legend_labels, title='Cluster', fontsize='xx-large', title_fontsize=20)

            # Hide the x-axis and y-axis tick labels
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            #save fig
            Utils.save_fig(plt, output_root_dir, output_fig_dir, model_name, model_id, output_fig_filename)
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)

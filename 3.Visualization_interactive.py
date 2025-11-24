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
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def save_3d_subphenotype_plot(x, y, z, cluster,
                              out_html="subphenotypes_3d.html",
                              axis_labels=("X","Y","Z"),
                              marker_size=1.8,          # << smaller dots
                              marker_opacity=0.35,
                              title="3D Subphenotypes"):
    """
    Saves a standalone interactive 3D HTML plot and returns nothing.
    x, y, z : 1D arrays/lists of equal length
    cluster : 1D array/list of cluster IDs (ints/strings)
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z); cluster = np.asarray(cluster)
    assert len(x) == len(y) == len(z) == len(cluster), "x, y, z, cluster must be same length"

    uniq = np.array(sorted(np.unique(cluster)))
    n_cluster = len(uniq)

    # Your palette (cycled if clusters > 8)
    base_colors = ['#673ca8','#C34f77','#96af77','#6491c7','#F38c16','#FA40B6','#A84972','#AC04BF']
    if n_cluster > len(base_colors):
        mul = (n_cluster + len(base_colors) - 1) // len(base_colors)
        colors = (base_colors * mul)[:n_cluster]
    else:
        colors = base_colors[:n_cluster]

    # Your label rule and color map
    labels_map = {c: f"Subphenotype {c+1}" for c in uniq}
    color_map  = {labels_map[c]: colors[i] for i, c in enumerate(uniq)}
    cat_order  = [labels_map[c] for c in uniq]

    df = pd.DataFrame({"x": x, "y": y, "z": z, "cluster": cluster})
    df["label"] = df["cluster"].map(labels_map)

    fig = px.scatter_3d(
        df, x="x", y="y", z="z",
        color="label",
        category_orders={"label": cat_order},
        color_discrete_map=color_map,
        hover_data={"x":":.3f", "y":":.3f", "z":":.3f", "label":True, "cluster":True},
        title=title,
        labels={"x": axis_labels[0], "y": axis_labels[1], "z": axis_labels[2], "label": "Subphenotype"}
    )
    fig.update_traces(
        marker=dict(
            size=marker_size,
            opacity=marker_opacity
        )
    )
    fig.update_layout(
        legend_title_text="Subphenotype",
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(out_html, include_plotlyjs="inline", full_html=True, auto_open=False)




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
            # assume you already have: x, y, cluster, node_size, n_cluster
            z = decom_results[:, 2]

            save_3d_subphenotype_plot(x, y, z, cluster, out_html=f"{output_root_dir}/{output_fig_dir}/{model_name}/{model_id}/interactive_3D_{method_name}.html")

    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)

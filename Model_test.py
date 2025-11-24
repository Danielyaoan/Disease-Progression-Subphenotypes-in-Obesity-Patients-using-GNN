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
import time
import random
from datetime import datetime
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print("Memory Allocated:", torch.cuda.memory_allocated(device) / (1024 ** 2), "MB")
    print("Memory Cached:", torch.cuda.memory_reserved(device) / (1024 ** 2), "MB")


class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=4, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
        


def execute(cfg, pfe):
    print("Loading parameters!")
    # loading all parameter
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    processed_dir = cfg.dataset_dir
    output_root_dir = cfg.output_root_dir
    output_features_dir = cfg.output_features_dir
    features_embedding_file_name = cfg.features_embedding_file_name
    output_log_dir = cfg.output_log_dir
    output_log_filename = cfg.output_log_filename

    path_dataset = os.path.join(root_dir, processed_dir, cfg.dataset_filename)
    path_adj_matrix = os.path.join(root_dir, processed_dir,cfg.adj_matrix_filename)
    # weight degree rev_sigmoid
    k = cfg.k
    # training settings
    train_test_ratio = cfg.patients_train_test_ratio
    batch_training = cfg.batch_training
    batch_size = cfg.batch_size
    batch_log_size = cfg.batch_log_size
    batch_model_save_number = cfg.batch_model_save_number
    
    alpha = cfg.alpha
    model_type = cfg.model_type  # could be 'GAT', 'GraphSAGE', 'DIGCN', 'DGCN'
    hidden_dim = cfg.model_params.hidden_dim
    output_embedding_dim = cfg.model_params.output_embedding_dim # embedding size
    num_hidden_layers = cfg.model_params.num_hidden_layers # at least 2
    dropout_prob = cfg.model_params.dropout_prob
    epochs = cfg.model_params.epochs
    learning_rate = cfg.model_params.learning_rate
    weight_decay = cfg.model_params.weight_decay
    optimizer_type = cfg.model_params.optimizer_type # could be 'Adam' or 'Sgd' else will be 'Adam'
    best_model_val_loss = None
    best_model_f1 = None
    gamma = cfg.gamma

    
    if model_type == 'GAT':
        heads = cfg.model_params.heads
    else:
        heads = 4 
    if model_type == 'Magnet':
        # K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        # q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        q = cfg.model_params.q
        K = cfg.model_params.K
        Magnet_activation = cfg.model_params.Magnet_activation
    else:
        q = 0.25
        K = 2
        Magnet_activation = True
        
    print(model_type, 'batch:', batch_training)
    # loading data
    print("Loading data!")
    features, label, edge_index, edge_weight,  raw_df = Utils.load_data(path_dataset, path_adj_matrix, k)
    PATID_list = raw_df['PATID']

    
    # get train/val subgraph data
    print("Spliting data!")
    train_data, val_data, test_data = Utils.split_data_by_PATID(PATID_list, train_test_ratio, features, label, edge_index, edge_weight)

    #Read the indices from the CSV
    read_indices = pd.read_csv(".../Graphmodel/GraphModeling/Utils/Sampling_index/test_indices.csv")["index"].tolist()
    # Select a subset of the features for fitting based on the read indices
    raw_df = raw_df.iloc[read_indices, : ]
    raw_df.reset_index(drop=True, inplace=True)
    
    data = test_data
                
    # Count the frequency of each unique label
    label_counts = torch.bincount(data.y)

    # Calculate the percentage of each label
    total_labels = data.y.size(0)
    print(total_labels)
    label_percentages = (label_counts / total_labels) * 100
    
    # Display the results
    for i, count in enumerate(label_counts):
        print(f"Label {i}: Count = {count}, Percentage = {label_percentages[i]:.2f}%")
                
    data.num_nodes = len(raw_df)
    data.num_classes = len(label.unique())

    #loading hyperparameters
    input_dim = data.num_features
    output_dim = data.num_classes

    #build model
    print("Loading model!")
    print(output_root_dir, models_dir, model_name)
    # continue load previous model 
    model = Utils.load_model(output_root_dir, models_dir, model_name, model_id+'/best_f1')
    
    #loading model and data to GPU
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create a NeighborSampler for evaluation
    loader = NeighborLoader(test_data, num_neighbors=[-1]*model.num_layers, batch_size=batch_size, shuffle=False)

    
    all_preds = []
    all_labels = []
    

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma = gamma)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    if model_type == 'Magnet':
        with torch.no_grad():
            for i, batch in enumerate(loader):

                
                # push every thing to device
                batch_size = batch.batch_size
                x_batch = batch.x.to(device)
                y_batch = batch.y.to(device)
                batch_edge_index = batch.edge_index.to(device)
                batch_edge_weight = batch.edge_weight.to(device)

                # Get embeddings and logits for this batch
                embeddings_batch, logits_batch = model(x_batch, x_batch, batch_edge_index, batch_edge_weight)
        
                # Compute loss for this batch
                loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
                total_loss += loss.item() * batch_size
        
                # Compute accuracy for this batch
                preds = logits_batch[:batch_size].argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch[:batch_size].cpu().tolist())

                
                total_correct += (preds == y_batch[:batch_size]).sum().item()

                # which should be all node in batch
                total_samples += batch_size
                

        

    elif model_type == 'GCN':
        with torch.no_grad():
            for i, batch in enumerate(loader):

    
                # push every thing to device
                batch_size = batch.batch_size
                x_batch = batch.x.to(device)
                y_batch = batch.y.to(device)
                batch_edge_index = batch.edge_index.to(device)
                batch_edge_weight = batch.edge_weight.to(device)
                

    
                # Get embeddings and logits for this batch
                embeddings_batch, logits_batch = model(x_batch, batch_edge_index, batch_edge_weight)
                
                # Compute loss for this batch
                loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
                total_loss += loss.item() * batch_size
        
                # Compute accuracy for this batch
                preds = logits_batch[:batch_size].argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch[:batch_size].cpu().tolist())
                

                
                total_correct += (preds == y_batch[:batch_size]).sum().item()

                # which should be all node in batch
                total_samples += batch_size
                
        

    else:
        with torch.no_grad():
            for i, batch in enumerate(loader):

                # push every thing to device
                batch_size = batch.batch_size
                x_batch = batch.x.to(device)
                y_batch = batch.y.to(device)
                batch_edge_index = batch.edge_index.to(device)

    
                # Get embeddings and logits for this batch
                embeddings_batch, logits_batch = model(x_batch, batch_edge_index)
                
                # Compute loss for this batch
                loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
                total_loss += loss.item() * batch_size
        
                # Compute accuracy for this batch
                preds = logits_batch[:batch_size].argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch[:batch_size].cpu().tolist())
            
                
                total_correct += (preds == y_batch[:batch_size]).sum().item()
                # which should be all node in batch
                total_samples += batch_size
                

                
    
    # Compute overall accuracy and loss
    overall_acc = total_correct / total_samples
    overall_loss = total_loss / total_samples
    
    print(f'On the whole dataset: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}')
    
    counts_preds = Counter(all_preds)
    counts_labels = Counter(all_labels)
    
    # Print the counts for each class in all_preds
    print("Counts in all_preds:")
    for class_label, count in counts_preds.items():
        print(f"Count of {class_label} in all_preds: {count}")
    
    # Print the counts for each class in all_labels
    print("\nCounts in all_labels:")
    for class_label, count in counts_labels.items():
        print(f"Count of {class_label} in all_labels: {count}")
        



    
    # Convert all_preds to a Pandas Series and add it to raw_df
    all_preds_series = pd.Series(all_preds)
    all_labels_series = pd.Series(all_labels)
    
    raw_df['all_preds'] = all_preds_series
    raw_df['all_labels'] = all_labels_series
    
    
    print('pred:\n',raw_df['all_preds'].value_counts())
    print('label:\n',raw_df['all_labels'].value_counts())
    print('true:\n',raw_df['BMI_DIFF'].value_counts())
    
    raw_df['all_preds'] = raw_df['all_preds'].astype(int)
    raw_df['all_labels'] = raw_df['all_labels'].astype(int)
    ### raw_df['current_status'] = raw_df['current_status'].astype(int)
    raw_df['BMI_DIFF'] = raw_df['BMI_DIFF'].astype(int)
    
    # Calculate the overall matching percentage
    overall_match_percentage = (len(raw_df[raw_df['BMI_DIFF'] == raw_df['all_preds']]) / len(raw_df))*100
    print(f"Overall percentage of rows where 'BMI_DIFF' matches 'all_preds': {overall_match_percentage:.2f}%")

    # Filter rows where 'current_status' is not equal to 'BMI_DIFF'
    ### filtered_df = raw_df[raw_df['current_status'] != raw_df['BMI_DIFF']]
    
    ### print('# status change:',len(filtered_df))
    
    ### print('pred:',filtered_df['all_preds'].value_counts())
    ### print('label:',filtered_df['all_labels'].value_counts())
    ### print('true:',filtered_df['BMI_DIFF'].value_counts())

    # Calculate the matching percentage in the filtered DataFrame
    ### filtered_match_percentage = (len(filtered_df[filtered_df['BMI_DIFF'] == filtered_df['all_preds']]) /len(filtered_df))*100
    ### print(f"Percentage of rows where 'current_status' != 'BMI_DIFF' and 'BMI_DIFF' matches 'all_preds': {filtered_match_percentage:.2f}%")

    # For the entire dataset
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    cm = confusion_matrix(all_labels, all_preds)
    
    # For the filtered dataset
    ### filtered_accuracy = accuracy_score(filtered_df['all_labels'], filtered_df['all_preds'])
    ### filtered_f1 = f1_score(filtered_df['all_labels'], filtered_df['all_preds'], average='macro')
    ### filtered_precision = precision_score(filtered_df['all_labels'], filtered_df['all_preds'], average='macro', zero_division=1)
    ### filtered_recall = recall_score(filtered_df['all_labels'], filtered_df['all_preds'], average='macro', zero_division=1)
    ### filtered_cm = confusion_matrix(filtered_df['all_labels'], filtered_df['all_preds'])
    
    # Binarize the output for multi-class AUROC
    n_classes = 4
    all_labels_binarized = label_binarize(all_labels, classes=[0, 1, 2, 3])
    all_preds_binarized = label_binarize(all_preds, classes=[0, 1, 2, 3])
    ### filtered_labels_binarized = label_binarize(filtered_df['all_labels'], classes=[0, 1, 2])
    ### filtered_preds_binarized = label_binarize(filtered_df['all_preds'], classes=[0, 1, 2])
    
    # Compute AUROC for each class and take the average, ensuring each class is represented
    auroc_scores = []
    ### filtered_auroc_scores = []
    
    for i in range(n_classes):
        if sum(all_labels_binarized[:, i]) > 0 and sum(all_preds_binarized[:, i]) > 0:
            auroc_scores.append(roc_auc_score(all_labels_binarized[:, i], all_preds_binarized[:, i]))
        ### if sum(filtered_labels_binarized[:, i]) > 0 and sum(filtered_preds_binarized[:, i]) > 0:
            ### filtered_auroc_scores.append(roc_auc_score(filtered_labels_binarized[:, i], filtered_preds_binarized[:, i]))
    
    average_auroc = sum(auroc_scores) / len(auroc_scores) if auroc_scores else 0
    ### filtered_average_auroc = sum(filtered_auroc_scores) / len(filtered_auroc_scores) if filtered_auroc_scores else 0

    
    # Sensitivity and Specificity for each class
    sensitivity_scores = []
    specificity_scores = []
    ### filtered_sensitivity_scores = []
    ### filtered_specificity_scores = []
    
    cm_size = cm.shape[0]
    ### filtered_cm_size = filtered_cm.shape[0]
    
    for i in range(cm_size):
        true_positive = cm[i, i]
        false_negative = sum(cm[i, j] for j in range(cm_size) if j != i)
        true_negative = sum(cm[j, k] for j in range(cm_size) for k in range(cm_size) if j != i and k != i)
        false_positive = sum(cm[j, i] for j in range(cm_size) if j != i)
    
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
    
    ### for i in range(filtered_cm_size):
        ### filtered_true_positive = filtered_cm[i, i]
        ### filtered_false_negative = sum(filtered_cm[i, j] for j in range(filtered_cm_size) if j != i)
        ### filtered_true_negative = sum(filtered_cm[j, k] for j in range(filtered_cm_size) for k in range(filtered_cm_size) if j != i and k != i)
        ### filtered_false_positive = sum(filtered_cm[j, i] for j in range(filtered_cm_size) if j != i)
    
        ### filtered_sensitivity = filtered_true_positive / (filtered_true_positive + filtered_false_negative) if (filtered_true_positive + filtered_false_negative) != 0 else 0
        ### filtered_specificity = filtered_true_negative / (filtered_true_negative + filtered_false_positive) if (filtered_true_negative + filtered_false_positive) != 0 else 0
    
        ### filtered_sensitivity_scores.append(filtered_sensitivity)
        ### filtered_specificity_scores.append(filtered_specificity)
    
    # Calculate the average sensitivity and specificity
    average_sensitivity = sum(sensitivity_scores) / cm_size
    average_specificity = sum(specificity_scores) / cm_size
    ### filtered_average_sensitivity = sum(filtered_sensitivity_scores) / filtered_cm_size
    ### filtered_average_specificity = sum(filtered_specificity_scores) / filtered_cm_size

    
    # Print all the metrics for the entire dataset
    print("Metrics for the entire dataset:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Average AUROC: {average_auroc}")
    print(f"Average Sensitivity: {average_sensitivity}")
    print(f"Average Specificity: {average_specificity}")
    
    # Print all the metrics for the filtered dataset
    ### print("\nMetrics for the filtered dataset:")
    ### print(f"Accuracy: {filtered_accuracy}")
    ### print(f"F1 Score: {filtered_f1}")
    ### print(f"Precision: {filtered_precision}")
    ### print(f"Recall: {filtered_recall}")
    ### print(f"Confusion Matrix:\n{filtered_cm}")
    ### print(f"Average AUROC: {filtered_average_auroc}")
    ### print(f"Filtered Average Sensitivity: {filtered_average_sensitivity}")
    ### print(f"Filtered Average Specificity: {filtered_average_specificity}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)

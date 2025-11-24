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
from scipy.stats import chi2_contingency

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def execute(cfg, pfe):
    # loading all parameter
    root_dir = cfg.root_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    processed_dir = cfg.dataset_dir
    output_root_dir = cfg.output_root_dir
    output_cluster_dir = cfg.output_cluster_dir
    cluster_filename = cfg.cluster_filename
    output_statistic_dir = cfg.output_statistic_dir
    statistic_filename = cfg.statistic_filename

    path_dataset = os.path.join(root_dir, processed_dir, cfg.dataset_filename)
    path_adj_matrix = os.path.join(root_dir, processed_dir,cfg.adj_matrix_filename)

    #cluster settings
    n_clusters = cfg.n_clusters
    distance_matrix = cfg.distance_matrix
    linkage = cfg.linkage


    # loading data
    # _, _, _, _, raw_df = Utils.load_data(path_dataset, path_adj_matrix, 1)
    
    file_dataset = open(path_dataset, 'rb')
    raw_df = pickle.load(file_dataset)
    
    #Read the indices from the CSV
    #read_indices = pd.read_csv(".../Graphmodel/GraphModeling/Utils/Sampling_index/3y/train_indices.csv")["index"].tolist()
    
    # Select a subset of the features for fitting based on the read indices
    #raw_df = raw_df.iloc[read_indices, : ]
    raw_df = raw_df.rename(columns={"BMI_DIFF": "status"})

    
    #print(raw_df.columns.tolist())

    # loading clustering info
    df_cluster = Utils.load_dataframe(output_root_dir, output_cluster_dir, model_name, model_id, cluster_filename)
    df_cluster['cluster_info'] = df_cluster['cluster_info'].astype(int)
    cluster = df_cluster['cluster_info'].values
    #cluster = cluster[read_indices]
    
    # combine the cluster results
    raw_df.iloc[:, 3:] = raw_df.iloc[:, 3:].astype(int)
    raw_df['cluster_info'] = cluster

    raw_df['status'] = raw_df['status'].astype(int)
    raw_df['cluster_info'] = raw_df['cluster_info'].astype(int)
    #raw_df['DAYS_SINCE_INDEX'] = raw_df['DAYS_SINCE_INDEX'].astype(int)
    
    
    #raw_df = pd.get_dummies(raw_df, columns=['Race_enthncity'], prefix='Race_enthncity')

    #if model_id == 'whole_dataset':
        # loading ADMIN_Date 
        #raw_df['ADMIT_DATE'] = raw_df.index.str[-10:]
    
    # mapping race value 
    # if '03' in raw_df.columns:
    #     raw_df = Utils.assign_race(raw_df)

    #calculate p value on different fearures
    p_value_df = pd.DataFrame(columns=['Characteristic', 'Total']+['Subphenotypes_' + str(i) for i in range(n_clusters)]+['P_value'])
    filtered_df_list = []
    for i in range(n_clusters):
        filtered_df = raw_df[raw_df['cluster_info'] == i]
        filtered_df_list.append(filtered_df)
        
    for feature in raw_df.columns:
        if str(feature) == 'PATID' or str(feature) == 'ENCID' or str(feature).startswith("DAYS_SINCE_INDEX"):
            continue
        
        if 'status' == str(feature):
            df = raw_df.copy()
            
            def average_difference(series):
                diff = series.diff().dropna()
                return diff.mean()
            
            
            df['with_bmi_diff_3'] = np.where(df['status'] == 3, 1, 0)
            
            columns_to_aggregate_0 = ['cluster_info']
            columns_to_aggregate_1 = ['status']
            columns_to_aggregate_2 = ['with_bmi_diff_3']

            # Create a dictionary of aggregation functions for each column
            agg_functions_0 = {col: 'first' for col in columns_to_aggregate_0}
            agg_functions_1 = {col: ['mean', average_difference] for col in columns_to_aggregate_1}
            agg_functions_2 = {col: 'max' for col in columns_to_aggregate_2}

            agg_functions_combined = {}
            agg_functions_combined.update(agg_functions_0)
            agg_functions_combined.update(agg_functions_1)
            agg_functions_combined.update(agg_functions_2)

            
            # Group by patient and apply the function
            status_times_df = df.groupby('PATID').agg(agg_functions_combined).reset_index()
            status_times_df.columns = ['PATID', 'cluster_info', 'avg_status', 'avg_status_difference', 'bmi_diff_3']
            
            
            
            
            def label_row(row):
                if row['avg_status']>=1.5 and row['bmi_diff_3']==1:                        # and row['avg_status_difference']>0:
                    return 'increase_with_3'
                elif row['avg_status']>=1.5:                                                # and row['avg_status_difference']>0:
                    return 'increase'
                elif row['avg_status']>=0.5:                                                # and row['avg_status_difference']>=0:
                    return 'steady'
                else:
                    return 'decrease'

            # Assuming your DataFrame is called df
            status_times_df['overall_situation'] = status_times_df.apply(label_row, axis=1)
            current_feature = 'overall_situation'
            feature_cat_list = status_times_df[current_feature].unique()
            for feature_cat in feature_cat_list:
                new_row_list = [current_feature + '_' +str(feature_cat)]
                category_counts = len(status_times_df[status_times_df[current_feature] == feature_cat])
                total_count = len(status_times_df[current_feature])

                percent = category_counts/total_count 
                new_row_list.append("{} ({:.1%})".format(category_counts , percent))

                for i in range(n_clusters):
                    filtered_df = status_times_df[status_times_df['cluster_info'] == i]
                    category_counts = len(filtered_df[filtered_df [current_feature] == feature_cat])
                    total_count = len(filtered_df[current_feature])

                    percent = category_counts/total_count 
                    new_row_list.append("{} ({:.1%})".format(category_counts , percent))

                #adding P value 
                p_df = pd.crosstab(status_times_df[current_feature], status_times_df['cluster_info'])
                c, p, dof, expected = chi2_contingency(p_df)
                new_row_list.append(round(p,4))

                #add list to df
                new_row_df = pd.DataFrame([new_row_list], columns=p_value_df.columns)
                p_value_df = pd.concat([p_value_df, new_row_df], ignore_index=True)
            

        elif feature == 'SEX_F' or feature == 'SEX_M':
            if feature == 'SEX_F':
                new_row_list = ['Female']
            else:
                new_row_list = ['Male']
            #for total
            temp_df = raw_df.copy()
            temp_df = temp_df.groupby('PATID')[[feature, 'cluster_info']].max().reset_index()
            
            sum = temp_df[feature].sum()
            percent = temp_df[feature].sum()/len(temp_df[feature])
            new_row_list.append("{} ({:.1%})".format(sum , percent))

            for i in range(n_clusters):
                filtered_df = filtered_df_list[i]
                temp_filtered_df = filtered_df.copy()
                temp_filtered_df = temp_filtered_df.groupby('PATID')[feature].max().reset_index()
                
                sum = temp_filtered_df[feature].sum()
                percent = sum/ len(temp_filtered_df)
                new_row_list.append("{:.0f} ({:.1%})".format(sum, percent))

            #adding P value     
            p_df = pd.crosstab(temp_df[feature], temp_df['cluster_info'])
            c, p, dof, expected = chi2_contingency(p_df)
            new_row_list.append(round(p,4))

            # add list to df
            new_row_df = pd.DataFrame([new_row_list], columns=p_value_df.columns)
            p_value_df = pd.concat([p_value_df, new_row_df], ignore_index=True)


        elif str(feature) == 'cluster_info':
            feature_cat_list = raw_df[feature].unique()
            for feature_cat in feature_cat_list:
                new_row_list = [str(feature) + '_' +str(feature_cat)]
                
                temp_df = raw_df.copy()
                temp_df = temp_df.groupby('PATID')['cluster_info'].max().reset_index()
                
                category_counts = len(temp_df[temp_df[feature] == feature_cat])
                total_count = len(temp_df[feature])

                percent = category_counts/total_count 
                new_row_list.append("{} ({:.1%})".format(category_counts , percent))

                for i in range(n_clusters):
                    filtered_df = filtered_df_list[i]
                    temp_filtered_df = filtered_df.copy()
                    temp_filtered_df = temp_filtered_df.groupby('PATID')['cluster_info'].max().reset_index()
                    
                    category_counts = len(temp_filtered_df[temp_filtered_df [feature] == feature_cat])
                    total_count = len(temp_filtered_df[feature])

                    percent = category_counts/total_count 
                    new_row_list.append("{} ({:.1%})".format(category_counts , percent))

                #adding P value 
                p_df = pd.crosstab(temp_df[feature], temp_df['cluster_info'])
                c, p, dof, expected = chi2_contingency(p_df)
                new_row_list.append(round(p,4))

                #add list to df
                new_row_df = pd.DataFrame([new_row_list], columns=p_value_df.columns)
                p_value_df = pd.concat([p_value_df, new_row_df], ignore_index=True)
        elif str(feature) not in ['Median BMI', 'Min BMI', 'CCI score', 'Count of obesity-related comorbidity', 'Avg AGE', 'Systolic blood pressure Max', 'Systolic blood pressure Median', 
                                  'Diastolic blood pressure Max', 'Diastolic blood pressure Median', 'Obesity_class Max', 'Obesity_class Min', 'Max unemployment rate', 'Max income level', 
                                  'Max median household income', 'Max education level', 'Max community food access level']:
            new_row_list =[str(feature)]
            #for total
            temp_df = raw_df.copy()
            temp_df = temp_df.groupby('PATID')[[feature, 'cluster_info']].max().reset_index()
            
            sum = temp_df[feature].sum()
            percent = temp_df[feature].sum()/len(temp_df[feature])
            new_row_list.append("{} ({:.1%})".format(sum , percent))

            for i in range(n_clusters):
                filtered_df = filtered_df_list[i]
                temp_filtered_df = filtered_df.copy()
                temp_filtered_df = temp_filtered_df.groupby('PATID')[feature].max().reset_index()
                
                sum = temp_filtered_df[feature].sum()
                percent = sum/ len(temp_filtered_df)
                new_row_list.append("{:.0f} ({:.1%})".format(sum, percent))

            # add list to df  
            p_df = pd.crosstab(temp_df[feature], temp_df['cluster_info'])
            c, p, dof, expected = chi2_contingency(p_df)
            new_row_list.append(round(p,4))

            # Convert the list to a Series object
            new_row_df = pd.DataFrame([new_row_list], columns=p_value_df.columns)
            p_value_df = pd.concat([p_value_df, new_row_df], ignore_index=True)
        else:
            new_row_list =[str(feature)]
            #for total
            avg = raw_df[feature].sum()/len(raw_df[feature])
            #percent = raw_df[feature].count()/len(raw_df[feature])
            new_row_list.append("{}".format(np.round(avg, 3)))

            for i in range(n_clusters):
                filtered_df = filtered_df_list[i]
                avg = filtered_df[feature].sum()/len(filtered_df)
                #percent = filtered_df[feature].count()/len(filtered_df)
                new_row_list.append("{}".format(np.round(avg, 3)))

            # add list to df  
            p_df = pd.crosstab(raw_df[feature], raw_df['cluster_info'])
            c, p, dof, expected = chi2_contingency(p_df)
            new_row_list.append(round(p,4))

            # Convert the list to a Series object
            new_row_df = pd.DataFrame([new_row_list], columns=p_value_df.columns)
            p_value_df = pd.concat([p_value_df, new_row_df], ignore_index=True)
    print(p_value_df.head())
    # save the statistic
    Utils.save_dataframe(p_value_df, output_root_dir, output_statistic_dir, model_name, model_id, statistic_filename)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)

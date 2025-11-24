import numpy as np
import pandas as pd
import os
import pickle

# loading raw features data
path_dataset = ".../graph_output_bmi_1&2_sub/total_enc_encode.pickle"

# loading files
file_dataset = open(path_dataset, 'rb')
df_features = pickle.load(file_dataset)

# List of columns to drop
columns_to_drop = ['PATID', 'BMI_DIFF']#, 'DAYS_SINCE_INDEX']'ADMIT_DATE', 'Min BMI'
# Drop the columns if they exist
raw_data = df_features.drop(columns=columns_to_drop, errors='ignore')


# Create a mapping of old names to new names
# problematic_features = [
#     col for col in df_features.columns 
#     if any(char in col for char in ['[', ']', '<']) and not col.startswith('AGE')
# ]
# rename_dict = {col: col.replace('[', '').replace(']', '').replace('<', '') for col in problematic_features}
# raw_data.rename(columns=rename_dict, inplace=True)

raw_data.reset_index(drop=True, inplace=True)


# Define the function to rename columns based on the provided conditions.
def rename_columns(df):
    # Create a dictionary for the new column names.
    new_column_names = {}
    # Iterate through each column in the dataframe.
    for column in df.columns:
        if '>' in column and '<' in column:
            df.rename(columns={column: column.replace('>', 'greater_').replace('<', 'smaller_')}, inplace=True)
        elif '>' in column:
            df.rename(columns={column: column.replace('>', 'greater_')}, inplace=True)
        elif '<' in column:
            df.rename(columns={column: column.replace('<', 'smaller_')}, inplace=True)
        elif '~' in column:
            df.rename(columns={column: column.replace('~', '_to_')}, inplace=True)

    # Rename the columns in the dataframe.
    # df.rename(columns=new_column_names, inplace=True)

    return df


# Apply the function to rename columns
raw_data = rename_columns(raw_data)


#loading cluster info
root_path = ".../Output_bmi_1&2_avg_min_sub/Cluster/"
output_filename = '.../Output_bmi_1&2_avg_min_sub/Cluster/whole_data_k_4.csv'
dataset = 'whole_dataset'
file_name = "Cluster_4.csv"


GAT_path = root_path + 'GAT' + "/"+ dataset +"/"+ file_name
GAT_data = pd.read_csv(GAT_path)[['ENCID', 'cluster_info']]
GCN_path = root_path + 'GCN' + "/"+ dataset +"/"+ file_name
GCN_data = pd.read_csv(GCN_path)[['ENCID', 'cluster_info']]
GraphSAGE_path = root_path + 'GraphSAGE' + "/"+ dataset +"/"+ file_name
GraphSAGE_data = pd.read_csv(GraphSAGE_path)[['ENCID', 'cluster_info']]
Magnet_path = root_path + 'Magnet' + "/"+ dataset +"/"+ file_name
Magnet_data = pd.read_csv(Magnet_path)[['ENCID', 'cluster_info']]

data = raw_data.merge(GAT_data, on='ENCID', how='left').rename(columns={'cluster_info': 'GAT_cluster'})
data = data.merge(GCN_data, on='ENCID', how='left').rename(columns={'cluster_info': 'GCN_cluster'})
data = data.merge(GraphSAGE_data, on='ENCID', how='left').rename(columns={'cluster_info': 'GraphSAGE_cluster'})
data = data.merge(Magnet_data, on='ENCID', how='left').rename(columns={'cluster_info': 'Magnet_cluster'})
data = data.drop(columns=['ENCID'], errors='ignore')
data.to_csv(output_filename, index = False)



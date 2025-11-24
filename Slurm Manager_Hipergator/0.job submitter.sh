#sbatch slurm_model_tuning.sh Settings/Tuning/tuning_GAT.json 
#sbatch slurm_model_tuning.sh Settings/Tuning/tuning_GCN.json 
#sbatch slurm_model_tuning.sh Settings/Tuning/tuning_GraphSAGE.json 
#sbatch slurm_model_tuning.sh Settings/Tuning/tuning_Magnet.json 

#sbatch slurm_Visualization.sh Settings/Visualization/Visualization_GAT.json
#sbatch slurm_Visualization.sh Settings/Visualization/Visualization_GCN.json  
#sbatch slurm_Visualization.sh Settings/Visualization/Visualization_GraphSAGE.json  
#sbatch slurm_Visualization.sh Settings/Visualization/Visualization_Magnet.json  

#sbatch slurm_Statistics.sh Settings/Statistic/statistic_GAT.json
#sbatch slurm_Statistics.sh Settings/Statistic/statistic_GCN.json  
#sbatch slurm_Statistics.sh Settings/Statistic/statistic_GraphSAGE.json  
#sbatch slurm_Statistics.sh Settings/Statistic/statistic_Magnet.json  

#sbatch slurm_model_test.sh Settings/Tuning/tuning_GAT.json
#sbatch slurm_model_test.sh Settings/Tuning/tuning_GCN.json 
#sbatch slurm_model_test.sh Settings/Tuning/tuning_GraphSAGE.json 
#sbatch slurm_model_test.sh Settings/Tuning/tuning_Magnet.json 

#sbatch slurm_Cluster.sh Settings/Clustering/clustering_GraphSAGE.json 
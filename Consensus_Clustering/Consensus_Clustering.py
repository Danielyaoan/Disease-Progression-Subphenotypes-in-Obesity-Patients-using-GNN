import numpy as np
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans  # Added import for TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

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
# Set the random seed for reproducibility
np.random.seed(42)

class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
        * NOTE: the class is to be instantiated with parameter `n_clusters`,
          and possess a `fit_predict` method, which is invoked on data.
        * L -> smallest number of clusters to try
        * K -> biggest number of clusters to try
        * H -> number of resamplings for each cluster number
        * resample_proportion -> percentage to sample
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters 
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in areas under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
      """

    def __init__(self, cluster, L, K, H, resample_proportion=0.5):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data, verbose=False):
        """
        Fits a consensus matrix for each number of clusters

        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in range(self.L_, self.K_):  # for each number of clusters
            i_ = k-self.L_
            if verbose:
                print("At k = %d, aka. iteration = %d" % (k, i_))
            for h in range(self.H_):  # resample H times
                if verbose:
                    print("\tAt resampling h = %d, (k = %d)" % (h, k))
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                index_mapping = np.array((Mh, resampled_indices)).T
                index_mapping = index_mapping[index_mapping[:, 0].argsort()]
                sorted_ = index_mapping[:, 0]
                id_clusts = index_mapping[:, 1]
                for i in range(k):  # for each cluster
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = id_clusts[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    # sometimes only one element is in a cluster (no combinations)
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                # increment counts
                ids_2 = np.array(list(combinations(resampled_indices, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is+1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1  # always with self
            Is.fill(0)  # reset counter
        self.Mk = Mk
        # fits areas under the CDFs
        self.Ak = np.zeros(self.K_-self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = np.sum(h*(b-a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format 
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)
            
            
#Read the indices from the CSV
read_indices = pd.read_csv("../.csv")["index"].tolist()
root_path = ""
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
print(f"Shape of {model} data: {data.shape}")

# Use these indices to get the sampled rows from the data
data = data[read_indices]

# Print the shape of the sampled data
print(f"Shape of sampled {model} data: {data.shape}")


resampled_indices = np.random.choice(range(data.shape[0]), size=int(data.shape[0]), replace=False)
data = data[resampled_indices, :]

chunk_size = int(0.1 * len(data))# Adjust based on your memory constraints
print('chunk size: ', chunk_size)

"""
def chunk_data(data, chunk_size):
    total_size = len(data)
    for i in range(0, total_size, chunk_size):
        if i + chunk_size > total_size:
            # If the end is reached, start the last chunk earlier to maintain the same size
            yield data[total_size - chunk_size:total_size]
            break
        else:
            yield data[i:i + chunk_size]
"""

data_list = []
total_size = len(data)
for i in range(0, total_size, chunk_size):
    if i + chunk_size > total_size:
        # If the end is reached, start the last chunk earlier to maintain the same size
        #data_list.append(data[total_size - chunk_size:total_size])
        break
    else:
        data_list.append(data[i:i + chunk_size])
        
# Once data_list is created, delete the original data to free up memory
del data


def process_chunk(data_chunk, cluster, L, K, H, resample_proportion):
    consensus_cluster = ConsensusCluster(cluster, L, K, H, resample_proportion)
    consensus_cluster.fit(data_chunk)
    return consensus_cluster.Mk
    


def aggregate_consensus_matrices(sum_matrix, new_matrix):
    # Aggregate by summing up the new matrix with the sum matrix
    return sum_matrix + new_matrix

# Initialize sum of consensus matrices
sum_Mk = None
# Initialize a counter for the number of chunks processed
count = 0

for data_chunk in data_list:
    Mk_chunk = process_chunk(data_chunk, KMeans, L=4, K=11, H=1, resample_proportion=1)
    count += 1  # Increment the count for each chunk
    print(count, len(data_chunk))
    if sum_Mk is None:
        sum_Mk = Mk_chunk
    else:
        sum_Mk = aggregate_consensus_matrices(sum_Mk, Mk_chunk)

# Compute the mean of the consensus matrices
overall_Mk = sum_Mk / count
"""
# Initialize sum of consensus matrices and count
sum_Mk = None
count = 0

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Submit tasks to the executor for each data chunk
    futures = [executor.submit(process_chunk, data_chunk, KMeans, L=4, K=9, H=1, resample_proportion=1) 
            for data_chunk in data_list]
#               for data_chunk in chunk_data(data, chunk_size)]
    
    for future in concurrent.futures.as_completed(futures):
        Mk_chunk = future.result()
        count += 1  # Increment the count for each completed chunk

        if sum_Mk is None:
            sum_Mk = Mk_chunk
        else:
            sum_Mk = aggregate_consensus_matrices(sum_Mk, Mk_chunk)

# Compute the mean of the consensus matrices
overall_Mk = sum_Mk / count
"""

def determine_final_bestK(overall_Mk, L, K):
    Ak = np.zeros(K-L)
    for i, m in enumerate(overall_Mk):
        hist, bins = np.histogram(m.ravel(), density=True)
        Ak[i] = np.sum(h*(b-a) for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
    
    deltaK = np.array([(Ak[i+1]-Ak[i])/Ak[i] for i in range(len(Ak)-1)])
    bestK = np.argmax(deltaK) + L
    return bestK

final_bestK = determine_final_bestK(overall_Mk, L=4, K=11)
print('Best K for the entire dataset:', final_bestK)



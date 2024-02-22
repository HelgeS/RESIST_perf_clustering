# %%
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import os.path as osp
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from common import (
    split_data,
    load_data,
    pearson_rank_distance_matrix,
    pearson_correlation,
    rank_difference_distance,
    wilcoxon_distance,
    kendalltau_distance,
)

# The plan for this file is to have two clusterings:
# 1. In the config feature space
# 2. In the input feature space
# The connection is via the distance function + ranking

# Notes
# a) Consistency: How different is feature-based clustering vs. rank-based clustering vs. perf-based clustering?
# b) Configuration recommendation
# - (x) Baseline: kNN: Query k nearest neighbours and let them vote on recommended configuration
# - Rank-based clustering: Cluster inputs via rank-based distance, look up closest cluster via input features and pick
# - Hierarchical clustering:
# - Metric: MAPE of configuration performance, Recall@k (open), Avg. rank of selected configuration
# c) Performance Prediction
# -

# %%
## Configuration
# Enter names of performance columns to consider
performances = ["elapsedtime"]


## Load and prepare data
perf_matrix, input_features, config_features = load_data(data_dir="../data/")
data_split = split_data(perf_matrix)
train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]

# This is a look up for performance measurements from inputname + configurationID
input_config_map = (
    perf_matrix[["inputname", "configurationID"] + performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
all_input_names = pd.Series(
    input_config_map.index.get_level_values("inputname").unique()
)
all_config_ids = pd.Series(
    input_config_map.index.get_level_values("configurationID").unique()
)

error_mape = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()).abs() / abs(x.min())
)
average_mape = error_mape.mean(axis=1)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)
average_ranks = rank_map.mean(axis=1)

## Create functions
# Some require global information from above


# This is a look up for performance measurements from inputname + configurationID
all_input_names = pd.Series(perf_matrix.inputname.sort_values().unique())
all_config_ids = pd.Series(perf_matrix.configurationID.sort_values().unique())

input_config_map = (
    data_split["train_data"][["inputname", "configurationID"] + performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)

measurements = input_config_map.values.reshape(
    (len(data_split["train_inp"]), len(data_split["train_cfg"]), len(performances))
)
len(all_input_names), len(all_config_ids), measurements.shape

# %%

### Rank-based distance matrix

# Distance matrix may not include negative values, but spearman can be negative
# distance_matrix = spearman_rank_distance(measurements)

# Input x Input distance matrix
# distance_matrix_inputs = rank_difference_distance(measurements)
distance_matrix_configs = rank_difference_distance(np.swapaxes(measurements, 0, 1))

model_dist = NearestNeighbors(metric="precomputed")
model_dist.fit(distance_matrix_configs[:, :, 0])

# %%
## Evaluation


# %%

## Dendrograms of different distance functions
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# %%
# Distance between measurements
reshaped_array = measurements.reshape(measurements.shape[0], -1)
model = AgglomerativeClustering(
    n_clusters=10, compute_distances=True, linkage="average"
)
model.fit(reshaped_array)
plot_dendrogram(model, truncate_mode="level", p=20, color_threshold=0.01)

# %%
# Rank-difference distance
distance_matrix_inputs = rank_difference_distance(measurements)
model = AgglomerativeClustering(
    distance_threshold=0.0,
    n_clusters=None,
    compute_distances=True,
    linkage="average",
    metric="precomputed",
)
model.fit(distance_matrix_inputs[:, :, 0])
plot_dendrogram(model, truncate_mode="level", p=20, color_threshold=0.01)

# %%
# Wilcoxon distance
dist_mat = wilcoxon_distance(measurements)
model = AgglomerativeClustering(
    distance_threshold=0.0,
    n_clusters=None,
    compute_distances=True,
    linkage="average",
    metric="precomputed",
)
model.fit(dist_mat[:, :, 0])
plot_dendrogram(model, truncate_mode="level", p=20, color_threshold=0.01)

# %%
# Kendalltau distance
dist_mat = kendalltau_distance(measurements)
model = AgglomerativeClustering(
    distance_threshold=0,
    n_clusters=None,
    compute_distances=True,
    linkage="average",
    metric="precomputed",
)
model.fit(dist_mat[:, :, 0])
plot_dendrogram(model, truncate_mode="level", p=20, color_threshold=0.01)

# %%

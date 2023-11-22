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
from common import split_data, load_data

performances = [
    "elapsedtime"  # we only consider this performance metric for the moment
]


# %%

# Data loading and preprocessing
# This is still redundant and mixes up Paul's original methods and my experimental code
perf_matrix, input_features, config_features = load_data()

# concat: input config performances

# We overwrite meas_matrix to only hold the values that are still in perf_matrix
# meas_matrix = perf_matrix[
#     ["configurationID", "inputname"] + config_columns_cont + performances
# ].join(pd.get_dummies(perf_matrix[config_columns_cat]))

# idx = compute_index(perf_matrix, nb_data)
# data_per_cfg = sort_data(meas_matrix, idx, nb_data)
# measures = cluster.extract_feature(data=data_per_cfg, nb_meas=nb_meas+1).iloc[:, 1:nb_meas]  # don't use size column
# measures = data_per_cfg[performances]
# index_interest = [0, 1, 2]
# feature_pts = cluster.create_feature_points(measures, nb_data, index_interest)

# This is a look up for performance measurements from inputname + configurationID
input_config_map = (
    perf_matrix[["inputname", "configurationID"] + performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
all_input_names = pd.Series(
    input_config_map.index.get_level_values("inputname").unique()
)
# input_config_map.index.get_level_values("inputname").unique().tolist()
all_config_ids = pd.Series(
    input_config_map.index.get_level_values("configurationID").unique()
)
# all_config_ids = (
#     input_config_map.index.get_level_values("configurationID").unique().tolist()
# )
measurements = input_config_map.values.reshape(
    (len(all_input_names), len(all_config_ids), len(performances))
)
len(all_input_names), len(all_config_ids), measurements.shape

# %%

## Split Data

data_split = split_data(perf_matrix)
train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]


# %%

## Prepare error look-up data frames

# scale to MAPE error (assuming minimization for all performance measures)
error_mape = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()).abs() / abs(x.min())
)
average_mape = error_mape.mean(axis=1)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)
average_ranks = rank_map.mean(axis=1)
# Look-up error as error_mape.loc[("Animation_1080P-01b3", 2)]["cpu"]
# Look-up rank as rank_map.loc[("Animation_1080P-01b3", 2)]["cpu"]


# %%

### Rank-based distance matrix

# Outline
# a) Consistency: How different is feature-based clustering vs. rank-based clustering vs. perf-based clustering?
# b) Configuration recommendation
# - Baseline: kNN: Query k nearest neighbours and let them vote on recommended configuration
# - Rank-based clustering: Cluster inputs via rank-based distance, look up closest cluster via input features and pick
# - Hierarchical clustering:
# - Metric: MAPE of configuration performance, Recall@k (open), Avg. rank of selected configuration
# c) Performance Prediction
# -

# %%

# Sketch
# distance_matrix = spearman_rank_distance(measurements)
# Distance matrix may not include negative values, but spearman can be negative
# model_dist = NearestNeighbors(metric="precomputed")
# model_dist.fit(distance_matrix[:, :, 0])

model_feat = NearestNeighbors()
model_feat.fit(input_features.loc[train_inp])


# %%

input_batch = input_features.iloc[:3]
evaluate_neighbour_ranks(input_batch, model_feat, topk=3)

# %%
## Evaluation


# %%

# x264
# 201 configurations, 1397 inputs without input properties
# 201 configurations, 1287 input with input properties

# 1. We need the ranks of the configs per input
# Redundant with earlier rank_map above
perf_matrix[["fps_rank", "kbs_rank"]] = (
    perf_matrix[["inputname", "fps", "kbs"]].groupby("inputname").rank()
)

mean_ranks = (
    perf_matrix[["configurationID", "fps_rank", "kbs_rank"]]
    .groupby("configurationID")
    .mean()
)


# %%
def distance_matrix_by_first_axis(array):
    reshaped_array = array.reshape(array.shape[0], -1)
    pairdist = cdist(reshaped_array, reshaped_array)
    return pairdist


def get_configuration_distances(measurements):
    return distance_matrix_by_first_axis(measurements)


def get_input_distances(measurements):
    return distance_matrix_by_first_axis(np.moveaxis(measurements, 0, 1))


# get_configuration_distances(measurements).mean(), get_input_distances(measurements).mean()

# %%
reshaped_array = measurements.reshape(measurements.shape[0], -1)
model = AgglomerativeClustering(
    n_clusters=10, compute_distances=False, linkage="average"
)
model.fit(reshaped_array)

# %%
model.n_clusters_

# %%
model.distances_.shape


# %%
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


plot_dendrogram(model, truncate_mode="level", p=20, color_threshold=0.01)


# %%

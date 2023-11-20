# %%
from main import load_all_csv, compute_index, sort_data
import cluster
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import datetime
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from sklearn.model_selection import train_test_split
import os.path as osp


def split_data(
    data, system, inputs_count, config_feat_cols, random_seed, test_size=0.2
):
    inputs = range(inputs_count[system])
    configs = pd.concat(
        (data[system, i][config_feat_cols[system]] for i in range(inputs_count[system]))
    ).drop_duplicates()

    split_test_size = test_size / 2
    train_inp, test_inp = train_test_split(
        inputs, test_size=split_test_size, random_state=random_seed
    )
    train_cfg, test_cfg = train_test_split(
        configs, test_size=split_test_size, random_state=random_seed
    )

    train_data = dict()
    test_data = dict()

    # test_cfg -> exclusive to test set, with all inputs
    for i in train_inp:
        train_data[system, i] = pd.merge(
            data[system, i], train_cfg, on=config_feat_cols[system], how="inner"
        )
        test_data[system, i] = pd.merge(
            data[system, i], test_cfg, on=config_feat_cols[system], how="inner"
        )
        assert (
            len(
                pd.merge(
                    train_data[system, i],
                    test_data[system, i],
                    on=config_feat_cols[system],
                    how="inner",
                )
            )
            == 0
        )

    # test_inp -> exclusive to test set, with all configs
    for i in test_inp:
        test_data[system, i] = data[system, i]

    return train_data, test_data, np.array(train_inp), train_cfg

def format_time(s):
    dtf = datetime.time.fromisoformat("00:0" + s.strip())
    return dtf.second + dtf.microsecond/1_000_000

# %%
nb_meas = 8  # this is how many measures were taken (i.e. how many measurement columns are in the dataframe)
perf_matrix, nb_data = load_all_csv("../data/res_ugc/", ext="csv", with_names=True)
# input_properties = pd.read_csv("../data/res_ugc_properties.csv")  # Does not match all inputs from perf_matrix?
# del input_properties["id"]
idx = compute_index(perf_matrix, nb_data)
data_per_cfg = sort_data(perf_matrix, idx, nb_data)
measures = cluster.extract_feature(data=data_per_cfg, nb_meas=nb_meas+1).iloc[:, 1:nb_meas]  # don't use size column
index_interest = [0,1,2]
feature_pts = cluster.create_feature_points(measures, nb_data, index_interest)
measure_names = list(measures.columns.values)

# %%

# We set aside 15% of configurations and 15% of inputs as test data
# This gives us 4 sets of data, of which we set 3 aside for testing
test_size = 0.15
train_cfg, test_cfg = train_test_split(perf_matrix["configurationID"].unique(), test_size=test_size)
train_inp, test_inp = train_test_split(perf_matrix["inputname"].unique(), test_size=test_size)
train_data = perf_matrix[perf_matrix["configurationID"].isin(train_cfg) & perf_matrix["inputname"].isin(train_inp)]
test_cfg_new = perf_matrix[perf_matrix["configurationID"].isin(test_cfg) & perf_matrix["inputname"].isin(train_inp)]
test_inp_new = perf_matrix[perf_matrix["configurationID"].isin(train_cfg) & perf_matrix["inputname"].isin(test_inp)]
test_both_new = perf_matrix[perf_matrix["configurationID"].isin(test_cfg) & perf_matrix["inputname"].isin(test_inp)]
assert test_cfg_new.shape[0] + test_inp_new.shape[0] + test_both_new.shape[0] + train_data.shape[0] == perf_matrix.shape[0] 

print(f"Training data: {100*train_data.shape[0]/perf_matrix.shape[0]:.2f}%")
print(f"Both new: {100*test_both_new.shape[0]/perf_matrix.shape[0]:.2f}%")
print(f"Config new: {100*test_cfg_new.shape[0]/perf_matrix.shape[0]:.2f}%")
print(f"Input new: {100*test_inp_new.shape[0]/perf_matrix.shape[0]:.2f}%")

# %%
input_config_map = data_per_cfg[["inputname", "configurationID"] + measure_names].sort_values(["inputname", "configurationID"]).set_index(["inputname", "configurationID"])
input_config_map["elapsedtime"] = input_config_map["elapsedtime"].apply(format_time)

all_input_names = {s: i for i, s in enumerate(input_config_map.index.get_level_values("inputname").unique())}
all_config_ids = {s: i for i, s in enumerate(input_config_map.index.get_level_values("configurationID").unique())}
measurements = input_config_map.values.reshape((len(all_input_names), len(all_config_ids), len(measure_names)))
len(all_input_names), len(all_config_ids), measurements.shape


# %%

# 201 configurations, 1397 inputs

# 1. We need the ranks of the configs per input
perf_matrix[["fps_rank", "kbs_rank"]] = perf_matrix[["inputname", "fps", "kbs"]].groupby("inputname").rank()

mean_ranks = perf_matrix[["configurationID", "fps_rank", "kbs_rank"]].groupby("configurationID").mean()

# Comparing the ranks between two inputs:
# a) Higher emphasis on first ranks (= best configurations)
# 
# Do we first rank each performance criterion individually and then calculate the distance over the individual rank comparisons?
# Metrics: weighted disagreement in config ids? (not meaningful due to large number), weighted distance in ranks?
# How would we rank all criteria without imposing an order?

# %%
def distance_matrix_by_first_axis(array):
    reshaped_array = array.reshape(array.shape[0], -1)
    pairwise_distances = cdist(reshaped_array, reshaped_array)
    return pairwise_distances

def get_configuration_distances(measurements):
    return distance_matrix_by_first_axis(measurements)

def get_input_distances(measurements):
    return distance_matrix_by_first_axis(np.moveaxis(measurements, 0, 1))

# %%
get_configuration_distances(measurements).mean(), get_input_distances(measurements).mean()

# %%
reshaped_array = measurements.reshape(measurements.shape[0], -1)
model = AgglomerativeClustering(n_clusters=10, compute_distances=False, linkage='average')
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



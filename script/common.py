import pandas as pd
from sklearn.model_selection import train_test_split
from main import load_all_csv
import numpy as np
import os
import datetime


def format_time(s):
    dtf = datetime.time.fromisoformat("00:0" + s.strip())
    return dtf.minute * 60 + dtf.second + dtf.microsecond / 1_000_000


input_columns = [
    "category",
    "resolution",
    "WIDTH",
    "HEIGHT",
    "SPATIAL_COMPLEXITY",
    "TEMPORAL_COMPLEXITY",
    "CHUNK_COMPLEXITY_VARIATION",
    "COLOR_COMPLEXITY",
]
input_columns_cat = ["category"]
input_columns_cont = [s for s in input_columns if s not in input_columns_cat]

config_columns = [
    "cabac",
    "ref",
    "deblock",
    "analyse",
    "me",
    "subme",
    "mixed_ref",
    "me_range",
    "trellis",
    "8x8dct",
    "fast_pskip",
    "chroma_qp_offset",
    "bframes",
    "b_pyramid",
    "b_adapt",
    "direct",
    "weightb",
    "open_gop",
    "weightp",
    "scenecut",
    "rc_lookahead",
    "mbtree",
    "qpmax",
    "aq-mode",
]
config_columns_cat = [
    "analyse",
    "me",
    "direct",
    "deblock",
    "b_adapt",
    "b_pyramid",
    "open_gop",
    "rc_lookahead",
    "scenecut",
    "weightb",
]
config_columns_cont = [s for s in config_columns if s not in config_columns_cat]

all_performances = [
    "rel_size",  # after preprocessing, before just `size` - min
    "usertime",
    "systemtime",
    "elapsedtime",
    "cpu",
    # "frames",  # irrelevant?
    "fps",  # max
    "kbs",
]


def load_data(data_dir="../data/"):
    meas_matrix, _ = load_all_csv(
        os.path.join(data_dir, "res_ugc"), ext="csv", with_names=True
    )
    meas_matrix["elapsedtime"] = meas_matrix["elapsedtime"].apply(format_time)
    input_properties = pd.read_csv(
        os.path.join(data_dir, "res_ugc_properties.csv")
    )  # Does not match all inputs from perf_matrix?
    del input_properties["id"]

    perf_matrix = pd.merge(
        meas_matrix, input_properties, left_on="inputname", right_on="name"
    )
    del perf_matrix["name"]
    perf_matrix["rel_size"] = perf_matrix["size"] / perf_matrix["ORIG_SIZE"]
    # perf_matrix["rel_size"] = np.log(perf_matrix["rel_size"])  # To scale value distribution more evenly
    # TODO rel_kbs too?

    input_features = (
        perf_matrix[["inputname"] + input_columns_cont]
        .join(pd.get_dummies(perf_matrix[input_columns_cat]))
        .set_index("inputname")
        .drop_duplicates()
    )

    config_features = (
        perf_matrix[["configurationID"] + config_columns_cont]
        .join(pd.get_dummies(perf_matrix[config_columns_cat]))
        .set_index("configurationID")
        .drop_duplicates()
    )

    return perf_matrix, input_features, config_features


def split_data(perf_matrix, test_size=0.15, verbose=True):
    # We set aside 15% of configurations and 15% of inputs as test data
    # This gives us 4 sets of data, of which we set 3 aside for testing
    train_cfg, test_cfg = train_test_split(
        perf_matrix["configurationID"].unique(), test_size=test_size
    )
    train_inp, test_inp = train_test_split(
        perf_matrix["inputname"].unique(), test_size=test_size
    )
    train_data = perf_matrix[
        perf_matrix["configurationID"].isin(train_cfg)
        & perf_matrix["inputname"].isin(train_inp)
    ]
    test_cfg_new = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        & perf_matrix["inputname"].isin(train_inp)
    ]
    test_inp_new = perf_matrix[
        perf_matrix["configurationID"].isin(train_cfg)
        & perf_matrix["inputname"].isin(test_inp)
    ]
    test_both_new = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        & perf_matrix["inputname"].isin(test_inp)
    ]
    assert (
        test_cfg_new.shape[0]
        + test_inp_new.shape[0]
        + test_both_new.shape[0]
        + train_data.shape[0]
        == perf_matrix.shape[0]
    )

    if verbose:
        print(f"Training data: {100*train_data.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Both new: {100*test_both_new.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Config new: {100*test_cfg_new.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Input new: {100*test_inp_new.shape[0]/perf_matrix.shape[0]:.2f}%")

    return {
        "train_cfg": train_cfg,
        "test_cfg": test_cfg,
        "train_inp": train_inp,
        "test_inp": test_inp,
        "train_data": train_data,
        "test_data_cfg_new": test_cfg_new,
        "test_data_inp_new": test_inp_new,
        "test_data_both_new": test_both_new,
    }


## Rank-based distance calculation via correlation


# Function to calculate Pearson correlation coefficient in a vectorized manner
def pearson_correlation(X, Y):
    mean_X = np.mean(X, axis=-1, keepdims=True)
    mean_Y = np.mean(Y, axis=-1, keepdims=True)
    numerator = np.sum((X - mean_X) * (Y - mean_Y), axis=-1)
    denominator = np.sqrt(
        np.sum((X - mean_X) ** 2, axis=-1) * np.sum((Y - mean_Y) ** 2, axis=-1)
    )
    return numerator / denominator


def spearman_rank_distance(measurements):
    # Vectorized spearmanr with multiple measurements

    # Breaks ties correctly by assigning same ranks, but potentially instable
    # TODO Should work correctly if we drop `frames` which has constant value
    # ranks = stats.rankdata(measurements, axis=1, method="min")

    # Makes individual ranks
    ranks = np.argsort(measurements, axis=1)

    # The ranks array is 3D (A, B, C), and we need to expand it to 4D for pairwise comparison in A,
    # while keeping C
    expanded_rank_X_3d = ranks[:, np.newaxis, :, :]  # Expanding for A dimension
    expanded_rank_Y_3d = ranks[np.newaxis, :, :, :]  # Expanding for A dimension

    A = ranks.shape[0]
    C = ranks.shape[2]

    # Initialize the Spearman correlation matrix for each C
    spearman_correlation_matrix_3d = np.empty((A, A, C))

    # Calculate Spearman correlation matrix for each C
    for c in range(C):
        spearman_correlation_matrix_3d[:, :, c] = pearson_correlation(
            expanded_rank_X_3d[:, :, :, c], expanded_rank_Y_3d[:, :, :, c]
        )

    return spearman_correlation_matrix_3d


def rank_difference_distance(measurements):
    ranks = np.argsort(measurements, axis=1)
    expanded_ranks = ranks[:, np.newaxis, :, :] - ranks[np.newaxis, :, :, :]

    # Calculate the absolute differences and sum along the B dimension
    vectorized_distance_matrix = np.sum(np.abs(expanded_ranks), axis=2)
    return vectorized_distance_matrix

# %%
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from scipy import stats
from common import load_data, split_data

# %%

## Configuration
# Enter names of performance columns to consider
performances = ["elapsedtime"]

# Number of nearest neighbours to consider
topk_values = (1, 3, 5, 10)


## Load and prepare data
perf_matrix, input_features, config_features = load_data(data_dir="data/")
data_split = split_data(perf_matrix)

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


def configurations_from_neighbour_ranks(neighbor_indices, topk=5):
    avg_rank_per_measure = (
        rank_map.loc[all_input_names.iloc[neighbor_indices]]
        .groupby("configurationID")
        .mean()
    )
    aggregated_ranks = avg_rank_per_measure.mean(axis=1)
    return aggregated_ranks.nsmallest(topk).index


def evaluate_neighbour_ranks(input_batch, model_feat, topk):
    knn_indices = model_feat.kneighbors(input_batch, return_distance=False)

    recommended_configuration_ranks = np.apply_along_axis(
        configurations_from_neighbour_ranks, 1, knn_indices, topk=topk
    )

    # Best rank of recommendations
    best_ranks = np.array(
        [
            average_ranks.loc[(inp, rck)].min()
            for inp, rck in zip(input_batch.index, recommended_configuration_ranks)
        ]
    )

    # MAPE error in performance for best recommendation
    best_mape = np.array(
        [
            average_mape.loc[(inp, rck)].min()
            for inp, rck in zip(input_batch.index, recommended_configuration_ranks)
        ]
    )

    return best_ranks, best_mape


# %%

## Actual Execution

for topk in topk_values:
    model_feat = NearestNeighbors(n_neighbors=topk)
    model_feat.fit(input_features.loc[data_split["train_inp"]])

    best_ranks, best_mape = evaluate_neighbour_ranks(
        input_features.loc[data_split["test_inp"]], model_feat, topk=topk
    )

    print(
        f"Avg. rank of {topk} best recommended configuration: {best_ranks.mean():.2f}"
    )
    print(f"Avg. MAPE of {topk} best recommended configuration: {best_mape.mean():.2f}")

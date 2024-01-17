import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from main import load_all_csv
from scipy import stats
from sklearn.model_selection import KFold, train_test_split


def load_data(system, data_dir="../data"):
    if system == "nodejs":
        raise NotImplementedError(
            "We don't support nodejs / it's missing the configurationID in the measurements"
        )

    metadata = json.load(open(os.path.join(data_dir, "metadata.json")))
    system_metadata = metadata[system]
    config_columns = system_metadata["config_columns"]
    config_columns_cat = system_metadata["config_columns_cat"]
    config_columns_cont = system_metadata["config_columns_cont"]
    # input_columns = system_metadata["input_columns"]
    input_columns_cat = system_metadata["input_columns_cat"]
    input_columns_cont = system_metadata["input_columns_cont"]
    performances = system_metadata["performances"]

    meas_matrix, _ = load_all_csv(
        os.path.join(data_dir, system), ext="csv", with_names=True
    )
    input_properties = pd.read_csv(
        os.path.join(data_dir, system, "others", "properties.csv"),
        dtype={"name": "object"},
    ).set_index("id")  # id needed?

    # Rename columns with same name in inputs/perf. prediction to avoid errors later
    # Affects imagemagick
    for c in input_properties.columns:
        if c in performances or c in config_columns:
            input_properties.rename(columns={c: f"inp_{c}"}, inplace=True)

    perf_matrix = pd.merge(
        meas_matrix, input_properties, left_on="inputname", right_on="name"
    ).sort_values(by=["inputname", "configurationID"])
    del perf_matrix["name"]

    # System-specific adjustments
    # TODO Could be made part of the preprocessed datasets
    if system == "x264":
        # perf_matrix["rel_size"] = perf_matrix["size"] / perf_matrix["ORIG_SIZE"]  # We have `kbs` which is a better alternative
        # perf_matrix["rel_size"] = np.log(perf_matrix["rel_size"])  # To scale value distribution more evenly
        perf_matrix["rel_kbs"] = perf_matrix["kbs"] / perf_matrix["ORIG_BITRATE"]
        perf_matrix["fps"] = -perf_matrix[
            "fps"
        ]  # fps is the only increasing performance measure

    input_features = perf_matrix[["inputname"] + input_columns_cont]
    if len(input_columns_cat) > 0:
        input_features = input_features.join(
            pd.get_dummies(perf_matrix[input_columns_cat])
        )

    input_features = input_features.set_index("inputname").drop_duplicates()

    config_features = perf_matrix[["configurationID"] + config_columns_cont]
    if len(config_columns_cat) > 0:
        config_features = config_features.join(
            pd.get_dummies(perf_matrix[config_columns_cat])
        )

    config_features = config_features.set_index("configurationID").drop_duplicates()

    return perf_matrix, input_features, config_features, performances


def split_data(perf_matrix, test_size=0.2, verbose=True, random_state=None):
    # We set aside 20% of configurations and 20% of inputs as test data
    # This gives us 4 sets of data, of which we set 3 aside for testing
    train_cfg, test_cfg = train_test_split(
        perf_matrix["configurationID"].unique(),
        test_size=test_size,
        random_state=random_state,
    )
    train_inp, test_inp = train_test_split(
        perf_matrix["inputname"].unique(),
        test_size=test_size,
        random_state=random_state,
    )
    train_cfg.sort()
    test_cfg.sort()
    train_inp.sort()
    test_inp.sort()
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


def split_data_cv(perf_matrix, splits=4, verbose=True, random_state=None):
    kf_inp = KFold(n_splits=splits, random_state=random_state, shuffle=True)
    kf_cfg = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    configuration_ids = perf_matrix["configurationID"].unique()
    inputnames = perf_matrix["inputname"].unique()

    for split_idx, (
        (train_cfg_idx, test_cfg_idx),
        (train_inp_idx, test_inp_idx),
    ) in enumerate(zip(kf_inp.split(configuration_ids), kf_cfg.split(inputnames))):
        train_cfg = configuration_ids[train_cfg_idx]
        test_cfg = configuration_ids[test_cfg_idx]
        train_inp = inputnames[train_inp_idx]
        test_inp = inputnames[test_inp_idx]
        train_cfg.sort()
        test_cfg.sort()
        train_inp.sort()
        test_inp.sort()
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

        yield {
            "split": split_idx,
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


def stat_distance(measurements, stats_fn):
    ranks = np.argsort(measurements, axis=1)
    A = ranks.shape[0]
    C = ranks.shape[2]

    distance_matrix = np.zeros((A, A, C))

    # There is no good vectorized version to apply,
    # therefore we loop over all dimensions...
    for c in range(C):
        for i in range(A):
            for j in range(i + 1, A):
                try:
                    res = stats_fn(ranks[i, :, c], ranks[j, :, c])
                    stat, p_value = res.statistic, res.pvalue

                    distance_matrix[i, j, c] = stat
                    distance_matrix[j, i, c] = stat
                except ValueError:
                    # Mark as NaN in case of any other ValueError
                    distance_matrix[i, j, c] = np.nan
                    distance_matrix[j, i, c] = np.nan

    return distance_matrix


def kendalltau_distance(measurements):
    # We clip negative values to 0, because we see them as different
    # We invert such that lower values indicate higher correspondence
    return 1 - (np.maximum(stat_distance(measurements, stats_fn=stats.kendalltau), 0))


def wilcoxon_distance(measurements):
    return stat_distance(measurements, stats_fn=stats.wilcoxon)


def top_k_closest_euclidean(emb1, emb2=None, k=5):
    if emb2 is None:
        distance = torch.cdist(emb1, emb1, p=2)
        distance.fill_diagonal_(distance.max() + 1)
    else:
        distance = torch.cdist(emb1, emb2, p=2)

    return torch.topk(distance, k, largest=False, dim=1).indices


def top_k_closest_cosine(emb1, emb2, k):
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)

    # Calculate cosine similarity (dot product of unit vectors)
    similarity = torch.mm(emb1_norm, emb2_norm.t())
    return torch.topk(similarity, k, largest=True, dim=1).indices


## Evaluation functions

# These functions work on both the embedding space and the original feature space.
# Requirements:
# - The input/config representation must be floating point torch tensor.
# - `rank_arr` and `regret_arr` must be (I, C) tensors mapping the input idx x config idx to the performance measure

# TODO Make functions for rank_arr and regret_arr or let them work on the dataframes directly
# TODO Allow multiple performance measures
# TODO Move evaluation functions to separate file once they are stable

# rank_map -> IxCxP matrix


def evaluate_ic(
    input_representation,
    config_representation,
    rank_arr,
    regret_arr,
    k,
    distance="euclidean",
):
    """Evaluation of the input-configuration mapping.

    For each input, we look-up the `k` closest configurations in the representation space.
    Among them we evaluate the best and average performance in terms and rank and regret.
    """
    # TODO Mapping from embeddings to correct inputs/configs
    # For each input, we query the k closest configurations
    # We determine their rank against the measured data and the regret
    # We return the best and the mean value
    if distance == "euclidean":
        top_cfg = top_k_closest_euclidean(
            input_representation, config_representation, k=k
        )
    elif distance == "cosine":
        top_cfg = top_k_closest_cosine(input_representation, config_representation, k=k)

    # Ranks
    cfg_ranks = (
        (torch.gather(rank_arr, 1, top_cfg).float() - 1) / rank_arr.shape[1] * 100
    )
    best_rank = cfg_ranks.min(axis=1)[0].mean()
    avg_rank = cfg_ranks.mean(axis=1).mean()

    # Regret
    cfg_regret = torch.gather(regret_arr, 1, top_cfg).float() * 100
    best_regret = cfg_regret.min(axis=1)[0].mean()
    avg_regret = cfg_regret.mean(axis=1).mean()

    return best_rank, avg_rank, best_regret, avg_regret


def top_k_closest_euclidean_with_masks(emb, query_mask, reference_mask, k):
    if query_mask is None:
        query_mask = torch.ones(emb.shape[0], dtype=bool, device=emb.device)

    if reference_mask is None:
        reference_mask = torch.ones(emb.shape[0], dtype=bool, device=emb.device)

    queries = emb[query_mask]  # e.g. test inputs
    references = emb[
        reference_mask
    ]  # e.g. the training data for which we have measurements

    distance = torch.cdist(queries, references, p=2)
    shared_items = query_mask & reference_mask

    if shared_items.any():
        qm = query_mask.cumsum(dim=0) - 1
        dm = reference_mask.cumsum(dim=0) - 1

        indices = [qm[shared_items], dm[shared_items]]
        distance[indices] = distance.max() + 1

    top_refs = torch.topk(distance, k=k, largest=False, dim=1).indices

    # Remap reference indices in top_inp to original indices before continuing
    ref_idx = torch.where(reference_mask)[0]
    top_refs = torch.stack([ref_idx[r] for r in top_refs])

    assert top_refs.shape == (
        emb.shape[0] if query_mask is None else query_mask.sum(),
        k,
    ), top_refs.shape

    return top_refs


def evaluate_ii(
    input_representation,
    rank_arr,
    regret_arr,
    n_neighbors,
    n_recs=[1, 3, 5],
    query_mask=None,
    reference_mask=None,
):
    """
    Evaluation of the input representations.

    For each query input, we look-up the `n_neighbors` closest inputs in the representation space.
    Each of these neighbors recommends their top `n_recs` configuration.
    We evaluate:
    - The best rank any of the recommendations achieves on the query inputs
    - The best regret any of the recommendations achieves on the query inputs
    - The ratio of configurations that are common in the recommendations.
    """
    top_inp = top_k_closest_euclidean_with_masks(
        input_representation,
        query_mask=query_mask,
        reference_mask=reference_mask,
        k=n_neighbors,
    )

    # Foreach close input
    max_r = np.max(n_recs)
    top_r_ranks_per_neighbor = []
    top_r_regret_per_neighbor = []
    for r in top_inp:
        top_r_ranks_per_neighbor.append(
            torch.topk(rank_arr[r, :], k=max_r, dim=1, largest=False).indices
        )
        top_r_regret_per_neighbor.append(
            torch.topk(regret_arr[r, :], k=max_r, dim=1, largest=False).values
        )

    top_r_ranks_per_neighbor = torch.stack(top_r_ranks_per_neighbor)
    top_r_regret_per_neighbor = torch.stack(top_r_regret_per_neighbor)

    share_ratios = torch.empty(len(n_recs))
    best_regret = torch.empty(len(n_recs))
    best_rank = torch.empty(len(n_recs))
    n_queries = top_inp.shape[0]

    for i, r in enumerate(n_recs):
        # Ix(k*r) -> r highest ranked configs * k neighbors
        reduced_top_r = top_r_ranks_per_neighbor[:, :, :r].reshape(n_queries, -1)

        # Look-up the regret of the recommended configs on the query input
        # Per input take the best regret and the average over all query inputs
        best_regret[i] = (
            torch.tensor(
                [regret_arr[j, cfgs].min() for j, cfgs in enumerate(reduced_top_r)]
            ).mean()
            * 100
        )

        best_rank[i] = (
            (
                torch.tensor(
                    [rank_arr[j, cfgs].min() for j, cfgs in enumerate(reduced_top_r)],
                    dtype=torch.float,
                ).mean()
                - 1
            )
            / rank_arr.shape[1]
            * 100
        )

        # We must have at least r x num configs unique elements
        count_offset = n_queries * r
        uniq_vals = torch.tensor([torch.unique(row).numel() for row in reduced_top_r])
        share_ratios[i] = (
            1
            - (torch.sum(uniq_vals) - count_offset)
            / (reduced_top_r.numel() - count_offset)
        ) * 100

    return best_rank, best_regret, share_ratios


def evaluate_cc(
    config_representation,
    rank_arr,
    regret_arr,
    n_neighbors,
    n_recs=[1, 3, 5],
    query_mask=None,
    reference_mask=None,
):
    """
    Evaluation of the configuration representations.

    For each configuration, we look-up the `n_neighbors` closest configurations in the representation space.
    We evaluate the stability of configurations by:
    - The shared number of r affected inputs from 0 (no shared inputs) to 1 (all r inputs shared)

    n_recs is a parameter for the stability of the configuration.
    """
    top_cfg = top_k_closest_euclidean_with_masks(
        config_representation,
        query_mask=query_mask,
        reference_mask=reference_mask,
        k=n_neighbors,
    )

    # Foreach close config
    max_r = np.max(n_recs)
    top_r_ranks_per_neighbor = []
    top_r_regret_per_neighbor = []
    for neighbors in top_cfg:
        top_r_ranks_per_neighbor.append(
            torch.topk(rank_arr[:, neighbors], k=max_r, dim=0, largest=False).indices
        )
        top_r_regret_per_neighbor.append(
            torch.topk(regret_arr[:, neighbors], k=max_r, dim=0, largest=False).values
        )

    top_r_ranks_per_neighbor = torch.stack(top_r_ranks_per_neighbor)
    top_r_regret_per_neighbor = torch.stack(top_r_regret_per_neighbor)

    share_ratios = torch.empty(len(n_recs))
    best_regret = torch.empty(len(n_recs))
    best_rank = torch.empty(len(n_recs))
    n_queries = top_cfg.shape[0]

    for i, r in enumerate(n_recs):
        # Cx(k*r) -> r highest ranked inputs * k neighbors
        reduced_top_r = top_r_ranks_per_neighbor[:, :r, :].reshape(n_queries, -1)

        # Look-up the regret of the recommended configs on the query input
        # Per input take the best regret and the average over all query inputs
        best_regret[i] = (
            torch.tensor(
                [regret_arr[inps, j].min() for j, inps in enumerate(reduced_top_r)]
            ).mean()
            * 100
        )

        best_rank[i] = (
            (
                torch.tensor(
                    [rank_arr[inps, j].min() for j, inps in enumerate(reduced_top_r)],
                    dtype=torch.float,
                ).mean()
                - 1
            )
            / rank_arr.shape[0]
            * 100
        )

        # We must have at least r x num inputs unique elements
        count_offset = n_queries * r
        uniq_vals = torch.tensor([torch.unique(row).numel() for row in reduced_top_r])
        share_ratios[i] = (
            1
            - (torch.sum(uniq_vals) - count_offset)
            / (reduced_top_r.numel() - count_offset)
        ) * 100

    return best_rank, best_regret, share_ratios


def prepare_result_df(results, topr_values, topk_values, extra_info={}):
    df = pd.DataFrame(results, columns=topr_values)
    df["k"] = topk_values
    df.set_index("k", inplace=True)
    df.columns = pd.MultiIndex.from_product([["r"], df.columns])

    for k, v in extra_info.items():
        df[k] = v

    return df

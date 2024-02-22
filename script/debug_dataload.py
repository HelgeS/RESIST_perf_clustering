# %%
import argparse
import copy
import datetime
import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (
    evaluate_ic,
    evaluate_ii,
    load_data,
    make_latex_tables,
    pearson_rank_distance_matrix,
    split_data,
    split_data_cv,
    evaluate_prediction,
    evaluate_retrieval,
)
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn

from representation import (
    ListNetLoss,
    ccc_cmp_fn,
    cii_cmp_fn,
    icc_cmp_fn,
    iii_cmp_fn,
    make_batch,
)

# %%
data_dir = "../data/"
input_properties_type = "tabular"
system = "gcc"
method = "embed"
dimensions = 8

(
    perf_matrix,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(
    system=system, data_dir=data_dir, input_properties_type=input_properties_type
)
performance = all_performances[0]

print(f"Loaded data for `{system}`")
print(f"perf_matrix:{perf_matrix.shape}")
print(f"input_features(before preprocessing):{input_features.shape}")
print(f"config_features(before preprocessing):{config_features.shape}")

# This covers both train and test data
input_config_map_all = (
    perf_matrix[["inputname", "configurationID"] + [performance]]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)

regret_map_all = input_config_map_all.groupby("inputname").transform(
    lambda x: ((x - x.min()) / (x.max() - x.min()))  # .fillna(0)
)

rank_map_all = input_config_map_all.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)


# %%

data_split = split_data(perf_matrix)

train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]
test_inp = data_split["test_inp"]
test_cfg = data_split["test_cfg"]
train_data = data_split["train_data"]

# This is a look up for performance measurements from inputname + configurationID
# It only covers the training data
input_config_map = (
    train_data[["inputname", "configurationID"] + [performance]]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)

regret_map = input_config_map.groupby("inputname").transform(
    lambda x: ((x - x.min()) / (x.max() - x.min()))  # .fillna(0)
)

# We create the rank of inputs for a configuration by ranking their (input-internal) regret
cfg_rank_map = regret_map.groupby("configurationID").transform(
    lambda x: stats.rankdata(x, method="min")
)


# Prepare and select training/test data according to random split
train_input_mask = input_features.index.isin(train_inp)
test_input_mask = input_features.index.isin(test_inp)

train_config_mask = config_features.index.isin(train_cfg)
test_config_mask = config_features.index.isin(test_cfg)

input_preprocessor.fit(input_features[train_input_mask])
config_preprocessor.fit(config_features[train_config_mask])

input_arr = torch.from_numpy(input_preprocessor.transform(input_features)).float()
config_arr = torch.from_numpy(config_preprocessor.transform(config_features)).float()

train_input_arr = input_arr[train_input_mask]
train_config_arr = config_arr[train_config_mask]

test_input_arr = input_arr[test_input_mask]
test_config_arr = config_arr[test_config_mask]

# %%


def rankNet(
    y_pred,
    y_true,
    padded_value_indicator=-1,
    weight_by_diff=False,
    weight_by_diff_powed=False,
):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    y_true[mask] = float("-inf")

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(
        itertools.product(range(y_true.shape[1]), repeat=2)
    )

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(
            pairs_true[:, :, 1], 2
        )
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


# %%

emb_size = dimensions
num_input_features = train_input_arr.shape[1]
num_config_features = train_config_arr.shape[1]
input_map = {s: i for i, s in enumerate(train_inp)}
config_map = {s: i for i, s in enumerate(train_cfg)}
batch_size = 1024

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.0
input_emb = nn.Sequential(
    nn.Linear(num_input_features, 64),
    nn.Dropout(p=dropout),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.Dropout(p=dropout),
    nn.ReLU(),
    nn.Linear(64, emb_size),
).to(device)
config_emb = nn.Sequential(
    nn.Linear(num_config_features, 64),
    nn.Dropout(p=dropout),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.Dropout(p=dropout),
    nn.ReLU(),
    nn.Linear(64, emb_size),
).to(device)

optimizer = torch.optim.AdamW(
    list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.0001
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True)

# lnloss = ListNetLoss()
lnloss = rankNet

# Early stopping
best_loss = np.inf
best_loss_iter = 0
best_logmsg = None
best_models = (None, None)
patience = 200

# For evaluation
rank_arr = torch.from_numpy(
    rank_map.reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performance)
    .values
).to(device)
regret_arr = torch.from_numpy(
    regret_map.reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performance)
    .values
).to(device)
rank_arr_cfg = regret_arr.T.argsort(dim=1)

train_input_arr = train_input_arr.to(device)
train_config_arr = train_config_arr.to(device)

total_loss = 0


inp_cfg_ranks = regret_arr.argsort(dim=-1).float()
# inp_cfg_ranks = inp_cfg_ranks / inp_cfg_ranks.max(dim=-1, keepdim=True).values
icr = stats.rankdata(inp_cfg_ranks.numpy(), method="max", axis=-1)

cfg_inp_ranks = regret_arr.T.argsort(dim=-1).float()
# cfg_inp_ranks = cfg_inp_ranks / cfg_inp_ranks.max(dim=-1, keepdim=True).values
cir = stats.rankdata(cfg_inp_ranks.numpy(), method="max", axis=-1)


## Configuration - Configuration
cfg_cfg_corr = pearson_rank_distance_matrix(
    np.expand_dims(
        regret_map.loc[(train_inp, train_cfg), :]
        .reset_index()
        .pivot_table(index="configurationID", columns="inputname", values=performance)
        .values,
        axis=-1,
    )
).squeeze(-1)
cfg_cfg_ranks = stats.rankdata(cfg_cfg_corr, method="max", axis=-1)
# TODO Needs -1?
cfg_cfg_ranks_tensor = torch.from_numpy(cfg_cfg_ranks - 1).float().to(device)

## Input - Input
inp_inp_corr = pearson_rank_distance_matrix(
    np.expand_dims(
        regret_map.loc[(train_inp, train_cfg), :]
        .reset_index()
        .pivot_table(index="inputname", columns="configurationID", values=performance)
        .values,
        axis=-1,
    )
).squeeze(-1)
inp_inp_ranks = stats.rankdata(inp_inp_corr, method="max", axis=-1)
# TODO Needs -1?
inp_inp_ranks_tensor = torch.from_numpy(inp_inp_ranks - 1).float().to(device)


### TEST DATA
# TODO rankdata on test split or on full dataset?
cfg_cfg_corr_t = pearson_rank_distance_matrix(
    np.expand_dims(
        regret_map_all.loc[(test_inp, test_cfg), :]
        .reset_index()
        .pivot_table(index="configurationID", columns="inputname", values=performance)
        .values,
        axis=-1,
    )
).squeeze(-1)

# Yo this is not a rank, it's the pearson correlation
inp_inp_corr_t = pearson_rank_distance_matrix(
    np.expand_dims(
        regret_map_all.loc[(test_inp, test_cfg), :]
        .reset_index()
        .pivot_table(index="inputname", columns="configurationID", values=performance)
        .values,
        axis=-1,
    )
).squeeze(-1)

iirt = stats.rankdata(inp_inp_corr_t, method="max", axis=-1)
ccrt = stats.rankdata(cfg_cfg_corr_t, method="max", axis=-1)


def get_icrt_cirt(regret_map_all):
    regret_arr_all = torch.from_numpy(
        regret_map_all.loc[(test_inp, test_cfg), :]
        .reset_index()
        .pivot_table(index="inputname", columns="configurationID", values=performance)
        .values
    ).to(device)
    test_inp_cfg_ranks = regret_arr_all.argsort(dim=-1).float()
    # test_inp_cfg_ranks = (
    #     test_inp_cfg_ranks / test_inp_cfg_ranks.max(dim=-1, keepdim=True).values
    # )
    icrt = stats.rankdata(test_inp_cfg_ranks.numpy(), method="max", axis=-1)

    test_cfg_inp_ranks = regret_arr_all.T.argsort(dim=-1).float()
    # test_cfg_inp_ranks = (
    #     test_cfg_inp_ranks / test_cfg_inp_ranks.max(dim=-1, keepdim=True).values
    # )
    cirt = stats.rankdata(test_cfg_inp_ranks.numpy(), method="max", axis=-1)
    return icrt, cirt


icrt, cirt = get_icrt_cirt(regret_map_all)


def eval(inp_emb, cfg_emb, iir, ccr, icr, cir):
    correct_inp_inp = np.mean(
        stats.rankdata(torch.cdist(inp_emb, inp_emb).numpy(), axis=-1) <= iir
    )
    correct_cfg_cfg = np.mean(
        stats.rankdata(torch.cdist(cfg_emb, cfg_emb).numpy(), axis=-1) <= ccr
    )
    dist_ic = torch.cdist(inp_emb, cfg_emb).numpy()
    correct_inp_cfg = np.mean(stats.rankdata(dist_ic, axis=-1) <= icr)
    correct_cfg_inp = np.mean(stats.rankdata(dist_ic.T, axis=-1) <= cir)
    return (correct_inp_inp, correct_cfg_cfg, correct_inp_cfg, correct_cfg_inp)


best_loss = 999
best_correct = 0

for iteration in range(10_000):
    inp_emb = F.normalize(input_emb(train_input_arr))
    cfg_emb = F.normalize(config_emb(train_config_arr))

    optimizer.zero_grad()
    
    loss = 0
    
    
    distmat_inp = torch.cdist(inp_emb, inp_emb)
    loss += lnloss(
        distmat_inp,
        inp_inp_ranks_tensor,
    )
    distmat_cfg = torch.cdist(cfg_emb, cfg_emb)
    loss += lnloss(
        distmat_cfg,
        cfg_cfg_ranks_tensor,
    )

    distmat = torch.cdist(inp_emb, cfg_emb)
    loss += lnloss(
        distmat,
        inp_cfg_ranks,  # [input_indices, :][:, config_indices],
    )
    loss += lnloss(
        distmat.T,
        cfg_inp_ranks,  # [config_indices, :][:, input_indices],
    )

    if loss < best_loss:
        best_loss = loss.detach().item()

        with torch.no_grad():
            (correct_inp_inp, correct_cfg_cfg, correct_inp_cfg, correct_cfg_inp) = eval(
                inp_emb, cfg_emb, inp_inp_ranks, cfg_cfg_ranks, icr, cir
            )
            avg_train = np.mean(
                (correct_inp_inp, correct_cfg_cfg, correct_inp_cfg, correct_cfg_inp)
            )

            inp_emb_test = F.normalize(input_emb(test_input_arr))
            cfg_emb_test = F.normalize(config_emb(test_config_arr))
            (
                torrect_inp_inp,
                tcorrect_cfg_cfg,
                tcorrect_inp_cfg,
                tcorrect_cfg_inp,
            ) = eval(inp_emb_test, cfg_emb_test, iirt, ccrt, icrt, cirt)
            avg_test = np.mean(
                (
                    torrect_inp_inp,
                    tcorrect_cfg_cfg,
                    tcorrect_inp_cfg,
                    tcorrect_cfg_inp,
                )
            )

        print(
            f"{iteration}\t{loss.item():.3f} | {avg_train:.3f} | {correct_inp_inp:.3f} | {correct_cfg_cfg:.3f} | {correct_inp_cfg:.3f} | {correct_cfg_inp:.3f}"
        )
        print(
            f"test\t\t{torrect_inp_inp:.3f} | {avg_test:.3f} | {tcorrect_cfg_cfg:.3f} | {tcorrect_inp_cfg:.3f} | {tcorrect_cfg_inp:.3f}"
        )
        if correct_inp_inp > 0.99 and correct_cfg_cfg > 0.99:
            break

    loss.backward()
    optimizer.step()

# %%

# q: inp, r: inp
# Two inputs should be closer if they are similarly affected by the same configurations
# I.e. is their configuration rank similar -> correlation
# TODO rank correlation
# Correlation distance matrix
# a) with one performance measure
# b) with multiple performance measures
# Correlation metrics: spearman, kendalltau, rank_difference

# input-input:

measurements = input_config_map.values.reshape(
    (len(data_split["train_inp"]), len(data_split["train_cfg"]), 1)
)


# TODO If we have more than one performance metric,
# we can calculate the level in the pareto front as a rank
# This reduces to simply ranking in the case of one performance metric (nice!)
ic_dist_mat = stats.rankdata(measurements, axis=1)
ci_dist_mat = stats.rankdata(measurements, axis=0).swapaxes(0, 1)

# spearman rank distance
ii_dist_mat = pearson_rank_distance_matrix(measurements)
cc_dist_mat = pearson_rank_distance_matrix(measurements.swapaxes(0, 1))


# %%
perf_predict = nn.Sequential(
    nn.Linear(2 * emb_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1),  # TODO Check with |P| outputs
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y_train = torch.from_numpy(
    scaler.fit_transform(train_data[all_performances[0]].values.reshape(-1, 1))
)

train_indices = np.vstack(
    (
        train_data.configurationID.apply(
            lambda s: np.where(train_cfg == s)[0].item()
        ).values,
        train_data.inputname.apply(lambda s: np.where(train_inp == s)[0].item()).values,
    )
).T

X_train = torch.hstack(
    (
        config_emb(train_config_arr[train_indices[:, 0]]),
        input_emb(train_input_arr[train_indices[:, 1]]),
    )
)
regr_loss = F.mse_loss(perf_predict(X_train), y_train)
# %%

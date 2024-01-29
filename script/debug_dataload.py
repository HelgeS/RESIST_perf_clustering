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
    split_data,
    split_data_cv,
    evaluate_prediction,
    evaluate_retrieval,
)
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn

from representation import ListNetLoss, make_batch

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
performance = all_performances[0:1]

print(f"Loaded data for `{system}`")
print(f"perf_matrix:{perf_matrix.shape}")
print(f"input_features(before preprocessing):{input_features.shape}")
print(f"config_features(before preprocessing):{config_features.shape}")

# This is a look up for performance measurements from inputname + configurationID
input_config_map = (
    perf_matrix[["inputname", "configurationID"] + performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)

regret_map = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)

# %%

rank_arr = torch.from_numpy(
    rank_map.reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performances[0])
    .values
)
regret_arr = torch.from_numpy(
    regret_map.reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performances[0])
    .values
)

data_split = split_data(perf_matrix)

train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]
test_inp = data_split["test_inp"]
test_cfg = data_split["test_cfg"]

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

# %%

error_regret = regret_map

emb_size = dimensions
num_input_features = train_input_arr.shape[1]
num_config_features = train_config_arr.shape[1]
input_map = {s: i for i, s in enumerate(train_inp)}
config_map = {s: i for i, s in enumerate(train_cfg)}
batch_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_emb = nn.Sequential(
    nn.Linear(num_input_features, 64),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(64, emb_size),
).to(device)
config_emb = nn.Sequential(
    nn.Linear(num_config_features, 64),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(64, emb_size),
).to(device)

# TODO Implement performance prediction from latent space
# perf_predict = nn.Sequential(
#     nn.Linear(2 * emb_size, 64),
#     nn.ReLU(),
#     nn.Linear(64, 1),  # TODO Check with |P| outputs
# )

optimizer = torch.optim.AdamW(
    list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.0003
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True)

lnloss = ListNetLoss()

# Early stopping
best_loss = np.inf
best_loss_iter = 0
best_logmsg = None
best_models = (None, None)
patience = 200

# TODO For our dataset size it is relatively cheap to calculate the embeddings for all inputs and configs.
# We can every few iterations update a full collection and collect the hardest triplets from it.
#
# with torch.no_grad():
#     emb_lookup = torch.empty(
#         (train_input_arr.shape[0] + train_config_arr.shape[0], emb_size)
#     )
#     emb_lookup[: train_input_arr.shape[0]] = input_emb(train_input_arr)
#     emb_lookup[train_input_arr.shape[0] :] = config_emb(train_config_arr)

# For evaluation
rank_arr = torch.from_numpy(
    rank_map.loc[(train_inp, train_cfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performance)
    .values
).to(device)
regret_arr = torch.from_numpy(
    error_regret.loc[(train_inp, train_cfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performance)
    .values
).to(device)
rank_arr_cfg = regret_arr.T.argsort(dim=1)

train_input_arr = train_input_arr.to(device)
train_config_arr = train_config_arr.to(device)

total_loss = 0

# %%

with torch.no_grad():
    emb_lookup = torch.empty(
        (train_input_arr.shape[0] + train_config_arr.shape[0], emb_size)
    )
    emb_lookup[: train_input_arr.shape[0]] = input_emb(train_input_arr)
    emb_lookup[train_input_arr.shape[0] :] = config_emb(train_config_arr)



# %%
for iteration in range(3):
    batch, input_indices, config_indices = make_batch(
        train_inp,
        train_cfg,
        batch_size,
        input_map=input_map,
        config_map=config_map,
        rank_map=rank_map,
    )
    batch = batch.reshape((-1, 2)).to(device)
    input_indices = input_indices.to(device)
    config_indices = config_indices.to(device)
    input_row = batch[:, 0] == 1
    assert (
        batch.shape[1] == 2
    ), "Make sure to reshape batch to two columns (type, index)"

    optimizer.zero_grad()
    embeddings = torch.empty((batch.shape[0], emb_size), device=device)
    embeddings[input_row] = input_emb(train_input_arr[batch[input_row, 1]])
    embeddings[~input_row] = config_emb(train_config_arr[batch[~input_row, 1]])
    loss = nn.functional.triplet_margin_loss(
        anchor=embeddings[0::3],
        positive=embeddings[1::3],
        negative=embeddings[2::3],
    )

    distmat = torch.cdist(embeddings[input_row], embeddings[~input_row])
    loss += 0.05 * lnloss(
        torch.argsort(distmat, dim=1).float(),
        rank_arr[input_indices, :][:, config_indices].float(),
    )
    loss += 0.05 * lnloss(
        torch.argsort(distmat.T, dim=1).float(),
        rank_arr_cfg[config_indices, :][:, input_indices].float(),
    )

    loss.backward()
    optimizer.step()
    total_loss += loss.cpu().item()

# %%

distmat = torch.cdist(embeddings[input_row], embeddings[~input_row])

# q: inp, r: cfg
# rankarr
loss += 0.05 * lnloss(
    torch.argsort(distmat, dim=1).float(),
    rank_arr[input_indices, :][:, config_indices].float(),
)

# q: cfg, r: inp
loss += 0.05 * lnloss(
    torch.argsort(distmat.T, dim=1).float(),
    rank_arr_cfg[config_indices, :][:, input_indices].float(),
)

# q: inp, r: inp
# Two inputs should be closer if they are similarly affected by the same configurations
# I.e. is their configuration rank similar -> correlation
# TODO rank correlation
# Correlation distance matrix
# a) with one performance measure
# b) with multiple performance measures
# Correlation metrics: spearman, kendalltau, rank_difference

# q: cfg, r: cfg
# TODO rank correlation

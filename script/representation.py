# %%
import copy
import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (
    evaluate_cc,
    evaluate_ic,
    evaluate_ii,
    load_data,
    prepare_result_df,
    split_data,
    split_data_cv,
)
from scipy import stats
from torch import nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Purpose of this file
# We learn a simple embedding of the input and configuration vectors
# Input_embed() Config_embed() → joint embedding space
# Metric learning on triplets with relative relevance from collected data
# (A, P, N) triplet → Anchor, Positive Example, Negative Example
# I-I-I triplet: Inputs are closer if they benefit from the same configurations
# C-C-C triplet: Configurations are closer if they show higher effects on the same inputs
# C-I-I triplet: A configuration is closer to the input it has a higher effect on
# I-C-C triplet: An input is closer to the configuration it has a higher effect on

# Dataset construction
# Lookup table: (C,I), (I,I), (C,C)

# At each batch we need n triplets
# We can precalculate all triplets and pick from them or sample on the fly
# Let's first iterate on the pre-generation

# %%
## Load and prepare data
data_dir = "../data"
system = "x264"

perf_matrix, input_features, config_features, all_performances = load_data(
    system=system, data_dir=data_dir
)
performances = ["rel_kbs"]

print(f"Loaded data {system}")
print(f"perf_matrix:{perf_matrix.shape}")
print(f"input_features:{input_features.shape}")
print(f"config_features:{config_features.shape}")

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

regret_map = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()).abs() / abs(x.min())
)
average_regret = regret_map.mean(axis=1)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)
average_ranks = rank_map.mean(axis=1)

# Correlation cache
correlation_file = os.path.join(data_dir, "x264_correlations.p")
if os.path.exists(correlation_file):
    corr_dict = pickle.load(open(correlation_file, "rb"))
    input_correlations = corr_dict["input_correlations"]
    config_correlations = corr_dict["config_correlations"]
else:
    input_correlations = None
    config_correlations = None

# %%

# We define four functions to rank two items compared to an anchor item

# TODO cmp_fn can only handle a single performance measure in `rank_map`


def iii_cmp_fn(inp1, inp2, inp3, rank_map=None, lookup=None):
    """Returns which of inp2 and inp3 is closer to inp1 in terms of rank correlation (pearson)."""
    if lookup is not None:
        i1i2 = lookup[tuple(sorted((inp1, inp2)))]
        i1i3 = lookup[tuple(sorted((inp1, inp3)))]
    elif rank_map is not None:
        i1i2 = rank_map.loc[inp1].corrwith(rank_map.loc[inp2])
        i1i3 = rank_map.loc[inp1].corrwith(rank_map.loc[inp3])
    else:
        raise Exception("Either `rank_map` or `lookup` must be provided.")

    if (i1i2 < i1i3).item():
        return ("iii", inp1, inp2, inp3)
    else:
        return ("iii", inp1, inp3, inp2)


def ccc_cmp_fn(cfg1, cfg2, cfg3, rank_map=None, lookup=None):
    """Returns which of cfg2 and cfg3 is closer to cfg1 in terms of rank correlation (pearson)."""
    if lookup is not None:
        c1c2 = lookup[tuple(sorted((cfg1, cfg2)))]
        c1c3 = lookup[tuple(sorted((cfg1, cfg3)))]
    elif rank_map is not None:
        c1c2 = rank_map.xs(cfg1, level=1).corrwith(rank_map.xs(cfg2, level=1))
        c1c3 = rank_map.xs(cfg1, level=1).corrwith(rank_map.xs(cfg3, level=1))
    else:
        raise Exception("Either `rank_map` or `lookup` must be provided.")

    if (c1c2 < c1c3).item():
        return ("ccc", cfg1, cfg2, cfg3)
    else:
        return ("ccc", cfg1, cfg3, cfg2)


def cii_cmp_fn(cfg, inp1, inp2, rank_map):
    ci1 = rank_map.loc[(inp1, cfg)]
    ci2 = rank_map.loc[(inp2, cfg)]

    if (ci1 < ci2).item():
        return ("cii", cfg, inp1, inp2)
    else:
        return ("cii", cfg, inp2, inp1)


def icc_cmp_fn(inp, cfg1, cfg2, rank_map):
    ic1 = rank_map.loc[(inp, cfg1)]
    ic2 = rank_map.loc[(inp, cfg2)]

    if (ic1 < ic2).item():
        return ("icc", inp, cfg1, cfg2)
    else:
        return ("icc", inp, cfg2, cfg1)


# %%

# Here the actual training setup starts

train_input_arr = torch.from_numpy(input_features.loc[train_inp].values).float()
train_config_arr = torch.from_numpy(config_features.loc[train_cfg].values).float()

input_map = {s: i for i, s in enumerate(train_inp)}
config_map = {s: i for i, s in enumerate(train_cfg)}

# %%


# TODO Make vectorized version that splits evenly between the tasks
def make_batch(
    inputs, configs, size, input_map, config_map, rank_map=None, lookup=None
):
    batch_idx = []
    input_indices = []
    config_indices = []
    for _ in range(size):
        task = np.random.choice(4)
        if task == 0:  # iii
            params = np.random.choice(inputs, size=3, replace=False)
            triplet = iii_cmp_fn(*params, rank_map=rank_map, lookup=lookup)
        elif task == 1:  # ccc
            params = np.random.choice(configs, size=3, replace=False)
            triplet = ccc_cmp_fn(*params, rank_map=rank_map, lookup=lookup)
        elif task == 2:  # icc
            inp = np.random.choice(inputs)
            cfgs = np.random.choice(configs, size=2, replace=False)
            triplet = icc_cmp_fn(inp, *cfgs, rank_map=rank_map)
        else:  # cii
            cfg = np.random.choice(configs)
            inps = np.random.choice(inputs, size=2, replace=False)
            triplet = cii_cmp_fn(cfg, *inps, rank_map=rank_map)

        t, a, p, n = triplet
        batch_idx.append(
            (
                t[0] == "i",
                input_map[a] if t[0] == "i" else config_map[a],
                t[1] == "i",
                input_map[p] if t[1] == "i" else config_map[p],
                t[2] == "i",
                input_map[n] if t[2] == "i" else config_map[n],
            )
        )
        if t[0] == "i":
            input_indices.append(input_map[a])
        else:
            config_indices.append(config_map[a])

        if t[1] == "i":
            input_indices.append(input_map[p])
        else:
            config_indices.append(config_map[p])

        if t[2] == "i":
            input_indices.append(input_map[n])
        else:
            config_indices.append(config_map[n])

    return (
        torch.tensor(batch_idx),
        torch.tensor(input_indices),
        torch.tensor(config_indices),
    )


def make_batch_v2(inputs, configs, size, rank_map=None, lookup=None):
    """This samples a set of `size` inputs + configs and constructs all possible triplets from them."""
    half_size = size // 2
    sampled_ttinp = np.random.choice(inputs, size=half_size, replace=False)
    sampled_ttcfg = np.random.choice(configs, size=half_size, replace=False)
    batch_idx = []

    # iii task
    for inp1, inp2, inp3 in itertools.combinations(sampled_ttinp, 3):
        batch_idx.append(iii_cmp_fn(inp1, inp2, inp3, rank_map=rank_map, lookup=lookup))

    # ccc task
    for cfg1, cfg2, cfg3 in itertools.combinations(sampled_ttcfg, 3):
        batch_idx.append(ccc_cmp_fn(cfg1, cfg2, cfg3, rank_map=rank_map, lookup=lookup))

    # icc task
    for inp in sampled_ttinp:
        for cfg1, cfg2 in itertools.combinations(sampled_ttcfg, 2):
            batch_idx.append(icc_cmp_fn(inp, cfg1, cfg2, rank_map=rank_map))

    # cii task
    for cfg in sampled_ttcfg:
        for inp1, inp2 in itertools.combinations(sampled_ttinp, 2):
            batch_idx.append(cii_cmp_fn(cfg, inp1, inp2, rank_map=rank_map))

    # Convert to indices and tensor
    batch_idx = [
        (
            t[0] == "i",
            input_map[a] if t[0] == "i" else config_map[a],
            t[1] == "i",
            input_map[p] if t[1] == "i" else config_map[p],
            t[2] == "i",
            input_map[n] if t[2] == "i" else config_map[n],
        )
        for t, a, p, n in batch_idx
    ]
    return torch.tensor(batch_idx)


def make_batch_v3(inputs, configs, size, rank_map=None, lookup=None):
    mask = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=bool)
    batch_mask = mask[np.random.choice(mask.shape[0], size=10, replace=True)]
    n_inputs = batch_mask.sum()
    n_configs = (~batch_mask).sum()

    # selected_inputs = np.random.choice(len(inputs), size=(n_inputs,), replace=False)
    # seleced_configs = np.random.choice(len(configs), size=(n_configs,), replace=False)

    batch_idx = torch.empty((size, 6), dtype=int)
    batch_idx[:, 0::2] = batch_mask
    batch_idx[:, 1::2][batch_mask] = torch.from_numpy()
    batch_idx[:, 1::2][~batch_mask] = torch.from_numpy()

    for i in range(size):
        a, p, n = batch_idx[i, 1::2]

        if (batch_idx[i, 0::2] == mask[0]).all():
            row = iii_cmp_fn(
                inputs[a], inputs[p], inputs[n], rank_map=rank_map, lookup=lookup
            )[1:]
        elif (batch_idx[i, 0::2] == mask[1]).all():
            row = ccc_cmp_fn(
                configs[a], configs[p], configs[n], rank_map=rank_map, lookup=lookup
            )[1:]
        elif (batch_idx[i, 0::2] == mask[2]).all():
            row = icc_cmp_fn(inputs[a], configs[p], configs[n], rank_map=rank_map)[1:]
        elif (batch_idx[i, 0::2] == mask[3]).all():
            row = cii_cmp_fn(configs[a], inputs[p], inputs[n], rank_map=rank_map)[1:]
        else:
            raise Exception("Something went wrong")

        batch_idx[:, 1::2] = row

    return batch_idx


# %%


class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, pred_scores, true_scores):
        """
        pred_scores: Tensor of predicted scores (batch_size x list_size)
        true_scores: Tensor of ground truth scores (batch_size x list_size)
        """
        # Convert scores to probabilities
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)

        # Compute cross-entropy loss
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=1).mean()


# %%
def train_model(
    train_inp,
    train_cfg,
    train_input_arr,
    train_config_arr,
    rank_map,
    error_regret,
    performance,
):
    num_input_features = train_input_arr.shape[1]
    num_config_features = train_config_arr.shape[1]
    input_map = {s: i for i, s in enumerate(train_inp)}
    config_map = {s: i for i, s in enumerate(train_cfg)}
    emb_size = 32
    batch_size = 2048

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
        list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )

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

    for iteration in range(1000):
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

        if iteration > 0 and iteration % 10 == 0:
            total_loss /= 10
            scheduler.step(total_loss)

            with torch.no_grad():
                inputembs = input_emb(train_input_arr)

                if inputembs.std(axis=0).mean() < 0.05:
                    print("Input embeddings collapsed")
                    break

                (
                    icc_best_rank,
                    icc_avg_rank,
                    icc_best_regret,
                    icc_avg_regret,
                ) = evaluate_ic(
                    inputembs,
                    config_emb(train_config_arr),
                    rank_arr,
                    regret_arr,
                    k=5,
                )
                iii_ranks, iii_regret, _ = evaluate_ii(
                    inputembs,
                    rank_arr,
                    regret_arr,
                    n_neighbors=5,
                    n_recs=[1, 3, 5, 15, 25],
                )
                logmsg = (
                    f"{iteration} "
                    + f"l:{total_loss:.3f} | "
                    + f"ic(rank(best:{icc_best_rank:.2f} "
                    + f"avg:{icc_avg_rank:.2f}) "
                    + f"regret(best:{icc_best_regret:.2f} "
                    + f"avg:{icc_avg_regret:.2f})) | "
                    + f"ii(rank({iii_ranks.numpy().round(2)} "
                    + f"regret({iii_regret.numpy().round(2)}) "
                )
                print(logmsg)

                if total_loss < best_loss:
                    best_loss_iter = iteration
                    best_loss = total_loss
                    best_logmsg = logmsg
                    best_models = (copy.deepcopy(input_emb), copy.deepcopy(config_emb))
                elif (iteration - best_loss_iter) >= patience:
                    print(
                        f"No loss improvement since {patience} iterations. Stop training."
                    )
                    break

                total_loss = 0

    print(best_logmsg)
    return best_models


# %%
# def evaluate_cross_validation():
dfs = []
random_seed = 59590  # 33154

# Number of nearest neighbours to consider
# Make multiples to allow better budget comparison
topk_values = (1, 3, 5, 15, 25)
topr_values = (1, 3, 5, 15, 25)

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

for data_split in split_data_cv(perf_matrix, random_state=random_seed):
    train_inp = data_split["train_inp"]
    train_cfg = data_split["train_cfg"]
    test_inp = data_split["test_inp"]
    test_cfg = data_split["test_cfg"]

    # Prepare and select training/test data according to random split
    input_arr = torch.from_numpy(input_features.values).float()
    config_arr = torch.from_numpy(config_features.values).float()
    train_input_mask = input_features.index.isin(train_inp)
    test_input_mask = input_features.index.isin(test_inp)

    train_config_mask = config_features.index.isin(train_cfg)
    test_config_mask = config_features.index.isin(test_cfg)

    train_input_arr = input_arr[train_input_mask]
    train_config_arr = config_arr[train_config_mask]

    if method == "embed":
        best_models = train_model(
            train_inp=train_inp,
            train_cfg=train_cfg,
            train_input_arr=train_input_arr,
            train_config_arr=train_config_arr,
            rank_map=rank_map,
            error_regret=regret_map,
            emb_size=dimensions,
            performance=performances[0],
        )
        input_emb, config_emb = best_models
        input_emb = input_emb.cpu()
        config_emb = config_emb.cpu()

        torch.save(
            {
                "input_emb": input_emb,
                "config_emb": config_emb,
                "train_inp": train_inp,
                "train_cfg": train_cfg,
                "test_inp": test_inp,
                "test_cfg": test_cfg,
                "split": data_split["split"],
            },
            open(f"representation_{data_split['split']}_s{random_seed}.p", "wb"),
        )

        input_embeddings = input_emb(input_arr)
        config_embeddings = config_emb(config_arr)
    elif method == "tsne":
        input_embeddings = TSNE(dimensions).fit_transform(input_arr)
        config_embeddings = TSNE(dimensions).fit_transform(config_arr)
    elif method == "pca":
        input_embeddings = PCA(dimensions).fit_transform(input_arr)
        config_embeddings = PCA(dimensions).fit_transform(config_arr)

    train_cc_rank = []
    train_cc_ratio = []
    train_cc_regret = []

    test_cc_rank = []
    test_cc_ratio = []
    test_cc_regret = []

    train_ii_rank = []
    train_ii_ratio = []
    train_ii_regret = []

    test_ii_rank = []
    test_ii_ratio = []
    test_ii_regret = []

    # Query: test data
    # Database: train data

    for topk in topk_values:
        train_cc = evaluate_cc(
            config_embeddings,
            rank_arr=rank_arr,
            regret_arr=regret_arr,
            n_neighbors=topk,
            n_recs=topr_values,
            query_mask=torch.from_numpy(train_config_mask),
            reference_mask=torch.from_numpy(train_config_mask),
        )
        train_cc_rank.append(train_cc[0].numpy())
        train_cc_regret.append(train_cc[1].numpy())
        train_cc_ratio.append(train_cc[2].numpy())

        test_cc = evaluate_cc(
            config_embeddings,
            rank_arr=rank_arr,
            regret_arr=regret_arr,
            n_neighbors=topk,
            n_recs=topr_values,
            query_mask=torch.from_numpy(test_config_mask),
            reference_mask=torch.from_numpy(train_config_mask),
        )
        test_cc_rank.append(test_cc[0].numpy())
        test_cc_regret.append(test_cc[1].numpy())
        test_cc_ratio.append(test_cc[2].numpy())

        train_ii = evaluate_ii(
            input_embeddings,
            rank_arr=rank_arr,
            regret_arr=regret_arr,
            n_neighbors=topk,
            n_recs=topr_values,
            query_mask=torch.from_numpy(train_input_mask),
            reference_mask=torch.from_numpy(train_input_mask),
        )
        train_ii_rank.append(train_ii[0].numpy())
        train_ii_regret.append(train_ii[1].numpy())
        train_ii_ratio.append(train_ii[2].numpy())

        test_ii = evaluate_ii(
            input_embeddings,
            rank_arr=rank_arr,
            regret_arr=regret_arr,
            n_neighbors=topk,
            n_recs=topr_values,
            query_mask=torch.from_numpy(test_input_mask),
            reference_mask=torch.from_numpy(train_input_mask),
        )
        test_ii_rank.append(test_ii[0].numpy())
        test_ii_regret.append(test_ii[1].numpy())
        test_ii_ratio.append(test_ii[2].numpy())

    dfs.append(
        prepare_result_df(
            train_cc_rank,
            topr_values,
            topk_values,
            {"metric": "rank", "mode": "cc", "split": "train"},
        )
    )
    dfs.append(
        prepare_result_df(
            train_cc_regret,
            topr_values,
            topk_values,
            {"metric": "regret", "mode": "cc", "split": "train"},
        )
    )
    dfs.append(
        prepare_result_df(
            train_cc_ratio,
            topr_values,
            topk_values,
            {"metric": "ratio", "mode": "cc", "split": "train"},
        )
    )

    dfs.append(
        prepare_result_df(
            test_cc_rank,
            topr_values,
            topk_values,
            {"metric": "rank", "mode": "cc", "split": "test"},
        )
    )
    dfs.append(
        prepare_result_df(
            test_cc_regret,
            topr_values,
            topk_values,
            {"metric": "regret", "mode": "cc", "split": "test"},
        )
    )
    dfs.append(
        prepare_result_df(
            test_cc_ratio,
            topr_values,
            topk_values,
            {"metric": "ratio", "mode": "cc", "split": "test"},
        )
    )

    dfs.append(
        prepare_result_df(
            train_ii_rank,
            topr_values,
            topk_values,
            {"metric": "rank", "mode": "ii", "split": "train"},
        )
    )
    dfs.append(
        prepare_result_df(
            train_ii_regret,
            topr_values,
            topk_values,
            {"metric": "regret", "mode": "ii", "split": "train"},
        )
    )
    dfs.append(
        prepare_result_df(
            train_ii_ratio,
            topr_values,
            topk_values,
            {"metric": "ratio", "mode": "ii", "split": "train"},
        )
    )

    dfs.append(
        prepare_result_df(
            test_ii_rank,
            topr_values,
            topk_values,
            {"metric": "rank", "mode": "ii", "split": "test"},
        )
    )
    dfs.append(
        prepare_result_df(
            test_ii_regret,
            topr_values,
            topk_values,
            {"metric": "regret", "mode": "ii", "split": "test"},
        )
    )
    dfs.append(
        prepare_result_df(
            test_ii_ratio,
            topr_values,
            topk_values,
            {"metric": "ratio", "mode": "ii", "split": "test"},
        )
    )

full_df = pd.concat(dfs)
full_df.groupby(["mode", "split", "metric", "k"]).mean()

# %%

dfmean = (
    full_df.reset_index()
    .groupby(["mode", "split", "metric", "k"], as_index=False)
    .mean()
)

# %%

m = pd.concat(
    (
        dfmean[
            (dfmean["mode"] == "ii")
            & (dfmean["split"] == "test")
            & (dfmean["metric"] == "rank")
        ]
        .drop(columns=["mode", "split", "metric"])
        .set_index("k"),
        dfmean[
            (dfmean["mode"] == "ii")
            & (dfmean["split"] == "test")
            & (dfmean["metric"] == "ratio")
        ]
        .drop(columns=["mode", "split", "metric"])
        .set_index("k"),
        dfmean[
            (dfmean["mode"] == "ii")
            & (dfmean["split"] == "test")
            & (dfmean["metric"] == "regret")
        ]
        .drop(columns=["mode", "split", "metric"])
        .set_index("k"),
    ),
    axis=1,
    keys=["rank", "ratio", "regret"],
)
print(m.to_latex(index=True, float_format="%.2f", na_rep="-", caption="Input-Input"))

# %%

m = pd.concat(
    (
        dfmean[
            (dfmean["mode"] == "cc")
            & (dfmean["split"] == "test")
            & (dfmean["metric"] == "rank")
        ]
        .drop(columns=["mode", "split", "metric"])
        .set_index("k"),
        dfmean[
            (dfmean["mode"] == "cc")
            & (dfmean["split"] == "test")
            & (dfmean["metric"] == "ratio")
        ]
        .drop(columns=["mode", "split", "metric"])
        .set_index("k"),
        dfmean[
            (dfmean["mode"] == "cc")
            & (dfmean["split"] == "test")
            & (dfmean["metric"] == "regret")
        ]
        .drop(columns=["mode", "split", "metric"])
        .set_index("k"),
    ),
    axis=1,
    keys=["rank", "ratio", "regret"],
)
print(
    m.to_latex(
        index=True,
        float_format="%.2f",
        na_rep="-",
        caption="Configuration-Configuration",
    )
)

# %%

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

td = data_split["train_data"]
indices = np.vstack(
    (
        td.configurationID.apply(lambda s: np.where(train_cfg == s)[0].item()).values,
        td.inputname.apply(lambda s: np.where(train_inp == s)[0].item()).values,
    )
).T

scaler = StandardScaler()
y = scaler.fit_transform(td[performances[0]].values.reshape(-1, 1))

with torch.no_grad():
    X = torch.concat(
        (
            input_emb(train_input_arr)[indices[:, 1]],
            config_emb(train_config_arr)[indices[:, 0]],
        ),
        dim=1,
    ).numpy()

m = MLPRegressor(hidden_layer_sizes=(64,), verbose=True)
m.fit(X, y.ravel())
print("Train score", m.score(X, y))
yp = m.predict(X)
print("Train MAPE", mean_absolute_percentage_error(y, yp))

# %%

X = torch.concat(
    (
        train_input_arr[indices[:, 1]],
        train_config_arr[indices[:, 0]],
    ),
    dim=1,
).numpy()
m = MLPRegressor(hidden_layer_sizes=(64,), verbose=True)
m.fit(X, y.ravel())
print("Train score", m.score(X, y))
yp = m.predict(X)
print("Train MAPE", mean_absolute_percentage_error(y, yp))
# %%

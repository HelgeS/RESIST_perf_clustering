# %%
import itertools

import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from common import (
    load_data,
    pearson_rank_distance_matrix,
    split_data,
    pareto_rank
)
from learning import rankNet
from scipy import stats
from torch import nn

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

import plotnine as p9

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

icm = (
    perf_matrix[["inputname", "configurationID"] + all_performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
icm["ranks"] = icm.groupby("inputname", group_keys=False).apply(pareto_rank)

(
    p9.ggplot(icm.loc[("2mm")], p9.aes(x="exec", y="ctime", color="ranks"))
    + p9.geom_point()
    + p9.scale_color_cmap(cmap_name="tab20")
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

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

regret_val = (
    regret_map.reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performance)
    .values
)

# Kleinster Wert - kleinster/bester Rank
# rankNet: Bester Rank - hoechster Wert
# Modify rankNet: Adjust scores by subtraction from max or divide by max
# (not sure which or if it does matter)

# TODO Easy function for the whole data preparation and ranking

icr = stats.rankdata(regret_val, method="max", axis=-1)
inp_cfg_ranks = torch.from_numpy(icr - 1).float().to(device)

cir = stats.rankdata(regret_val.T, method="max", axis=-1)
cfg_inp_ranks = torch.from_numpy(cir - 1).float().to(device)


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
cfg_cfg_ranks_tensor = torch.from_numpy(cfg_cfg_ranks).float().to(device)

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
inp_inp_ranks_tensor = torch.from_numpy(inp_inp_ranks).float().to(device)


### TEST DATA
# TODO rankdata on test split or on full dataset?
cfg_cfg_corr_t = pearson_rank_distance_matrix(
    np.expand_dims(
        regret_map_all.reset_index()
        .pivot_table(index="configurationID", columns="inputname", values=performance)
        .values,
        axis=-1,
    )
).squeeze(-1)

# Yo this is not a rank, it's the pearson correlation
inp_inp_corr_t = pearson_rank_distance_matrix(
    np.expand_dims(
        regret_map_all.reset_index()
        .pivot_table(index="inputname", columns="configurationID", values=performance)
        .values,
        axis=-1,
    )
).squeeze(-1)

iirt = stats.rankdata(inp_inp_corr_t, method="max", axis=-1)
ccrt = stats.rankdata(cfg_cfg_corr_t, method="max", axis=-1)


def get_icrt_cirt(regret_map_all):
    regret_arr_all = (
        regret_map_all.reset_index()
        .pivot_table(index="inputname", columns="configurationID", values=performance)
        .values
    )

    icrt = stats.rankdata(regret_arr_all, method="max", axis=-1)
    cirt = stats.rankdata(regret_arr_all.T, method="max", axis=-1)

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


# %%


def train_func(
    train_input_arr,
    train_config_arr,
    input_arr,
    config_arr,
    emb_size,
    device,
    iterations,
    dropout=0.0,
    lr=0.0001,
    hidden_dim=64,
    do_eval=True,
    do_normalize=True,
    verbose=True,
):
    # iterations = 5000
    # emb_size = 16
    # device = device
    # dropout = 0.0
    # lr = 0.0003
    # hidden_dim = 64
    # do_eval = True
    # do_normalize = True

    num_input_features = train_input_arr.shape[1]
    num_config_features = train_config_arr.shape[1]

    input_emb = nn.Sequential(
        nn.Linear(num_input_features, hidden_dim),
        nn.Dropout(p=dropout),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Dropout(p=dropout),
        nn.ReLU(),
        nn.Linear(hidden_dim, emb_size),
    ).to(device)
    config_emb = nn.Sequential(
        nn.Linear(num_config_features, hidden_dim),
        nn.Dropout(p=dropout),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Dropout(p=dropout),
        nn.ReLU(),
        nn.Linear(hidden_dim, emb_size),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(input_emb.parameters()) + list(config_emb.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )

    # lnloss = ListNetLoss()
    lnloss = partial(
        rankNet, weight_by_diff=False, weight_by_diff_powed=False, max_subtract=True
    )
    lnloss = rankNet  # seems to work much better, but slower
    # does a form of triplet/pair mining internally

    # Early stopping
    best_loss = np.inf
    best_test_score = 0

    train_input_arr = train_input_arr.to(device)
    train_config_arr = train_config_arr.to(device)

    for iteration in range(iterations):
        inp_emb = input_emb(train_input_arr)
        cfg_emb = config_emb(train_config_arr)  # .detach()

        if do_normalize:
            inp_emb = F.normalize(inp_emb)
            cfg_emb = F.normalize(cfg_emb)

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

            if do_eval:
                with torch.no_grad():
                    (
                        correct_inp_inp,
                        correct_cfg_cfg,
                        correct_inp_cfg,
                        correct_cfg_inp,
                    ) = eval(inp_emb, cfg_emb, inp_inp_ranks, cfg_cfg_ranks, icr, cir)
                    avg_train = np.mean(
                        (
                            correct_inp_inp,
                            correct_cfg_cfg,
                            correct_inp_cfg,
                            correct_cfg_inp,
                        )
                    )

                    inp_emb_test = input_emb(input_arr)
                    cfg_emb_test = config_emb(config_arr)

                    if do_normalize:
                        inp_emb_test = F.normalize(inp_emb_test)
                        cfg_emb_test = F.normalize(cfg_emb_test)

                    # TODO Technically we only care about the ranking of the test inputs/configurations, not all of them
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
                    train.report(
                        {
                            "score": avg_test,
                            "train_score": avg_train,
                            "loss": best_loss,
                            "iteration": iteration,
                        }
                    )
                    if avg_test > best_test_score:
                        best_test_score = avg_test

                if verbose:
                    print(
                        f"{iteration}\t{loss.item():.3f} | {avg_train:.3f} | {correct_inp_inp:.3f} | {correct_cfg_cfg:.3f} | {correct_inp_cfg:.3f} | {correct_cfg_inp:.3f}"
                    )
                    print(
                        f"test\t\t{avg_test:.3f} | {torrect_inp_inp:.3f} | {tcorrect_cfg_cfg:.3f} | {tcorrect_inp_cfg:.3f} | {tcorrect_cfg_inp:.3f}"
                    )
                if correct_inp_inp > 0.99 and correct_cfg_cfg > 0.99:
                    break

        loss.backward()
        optimizer.step()

    train.report({"score": best_test_score, "loss": best_loss, "iteration": iteration})
    return best_loss, best_test_score


# train(
#     train_input_arr,
#     train_config_arr,
#     input_arr,
#     config_arr,
#     iterations=1000,
#     emb_size=16,
#     device=device,
#     dropout=0.0,
#     lr=0.0003,
#     hidden_dim=64,
#     do_eval=True,
# )


def tune_func(config):
    train_func(
        train_input_arr,
        train_config_arr,
        input_arr,
        config_arr,
        iterations=2500,
        emb_size=config["emb_size"],
        device=device,
        dropout=config["dropout"],
        lr=config["lr"],
        hidden_dim=config["hidden_dim"],
        do_normalize=config["do_normalize"],
        do_eval=True,
        verbose=False,
    )


search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "dropout": tune.uniform(0.0, 0.5),
    "emb_size": tune.choice([8, 16, 32, 64, 128, 256]),
    "hidden_dim": tune.choice([8, 16, 32, 64, 128, 256]),
    "max_subtract": tune.choice([True, False]),
    "do_normalize": tune.choice([True, False]),
}
algo = OptunaSearch()
tuner = tune.Tuner(
    tune_func,
    tune_config=tune.TuneConfig(
        metric="score", mode="max", search_alg=algo, num_samples=100
    ),
    param_space=search_space,
    run_config=train.RunConfig(
        stop={"training_iteration": 1000},
    ),
)
tuner.fit()

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
# perf_predict = nn.Sequential(
#     nn.Linear(2 * emb_size, 64),
#     nn.ReLU(),
#     nn.Linear(64, 1),  # TODO Check with |P| outputs
# )
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# y_train = torch.from_numpy(
#     scaler.fit_transform(train_data[all_performances[0]].values.reshape(-1, 1))
# )

# train_indices = np.vstack(
#     (
#         train_data.configurationID.apply(
#             lambda s: np.where(train_cfg == s)[0].item()
#         ).values,
#         train_data.inputname.apply(lambda s: np.where(train_inp == s)[0].item()).values,
#     )
# ).T

# X_train = torch.hstack(
#     (
#         config_emb(train_config_arr[train_indices[:, 0]]),
#         input_emb(train_input_arr[train_indices[:, 1]]),
#     )
# )
# regr_loss = F.mse_loss(perf_predict(X_train), y_train)
# %%

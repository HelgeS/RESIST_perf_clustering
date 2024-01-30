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
    split_data_cv,
    evaluate_prediction,
    evaluate_retrieval,
)
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn


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
            cfgs = np.random.choice(
                rank_map.xs(inp, level=0).index, size=2, replace=False
            )
            triplet = icc_cmp_fn(inp, *cfgs, rank_map=rank_map)
        else:  # cii
            cfg = np.random.choice(configs)
            inps = np.random.choice(
                rank_map.xs(cfg, level=1).index, size=2, replace=False
            )
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


def make_batch_v2(
    inputs, configs, size, input_map, config_map, rank_map=None, lookup=None
):
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
    # n_inputs = batch_mask.sum()
    # n_configs = (~batch_mask).sum()

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


class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, pred_scores, true_scores):
        """
        pred_scores: Tensor of predicted scores (batch_size x list_size)
        true_scores: Tensor of ground truth scores (batch_size x list_size)
        """

        true_nan = torch.isnan(true_scores)
        true_scores[true_nan] = float('-inf')
        pred_scores[true_nan] = float('-inf')

        if true_scores.max() > 1:
            true_scores = true_scores/true_scores.max()

        if pred_scores.max() > 1:
            pred_scores = pred_scores/pred_scores.max()

        # Convert scores to probabilities
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)

        # Compute cross-entropy loss
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=1).mean()


def train_model(
    train_inp,
    train_cfg,
    train_input_arr,
    train_config_arr,
    rank_map,
    error_regret,
    performance,
    emb_size,
    epochs,
):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )

    lnloss = ListNetLoss()

    # Early stopping
    best_loss = np.inf
    best_loss_iter = 0
    best_logmsg = None
    best_models = (copy.deepcopy(input_emb), copy.deepcopy(config_emb))
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

    for iteration in range(epochs):
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

        embeddings = F.normalize(embeddings, dim=1)

        loss = nn.functional.triplet_margin_loss(
            anchor=embeddings[0::3],
            positive=embeddings[1::3],
            negative=embeddings[2::3],
        )

        # TODO Add alternative ranking losses that prioritize top ranks
        # TODO Make lnlos weight a parameter + hpsearch
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


def evaluate_cv(
    method,
    dimensions,
    epochs,
    result_dir,
    perf_matrix,
    input_features,
    config_features,
    input_preprocessor,
    config_preprocessor,
    performances,
    random_seed=59590,
    topk_values=(1, 3, 5, 15, 25),
    topr_values=(1, 3, 5, 15, 25),
):
    dfs = []
    mape = []

    torch.random.seed(random_seed)
    np.random.seed(random_seed)

    for data_split in split_data_cv(perf_matrix, random_state=random_seed):
        train_inp = data_split["train_inp"]
        train_cfg = data_split["train_cfg"]
        test_inp = data_split["test_inp"]
        test_cfg = data_split["test_cfg"]
        train_data = data_split["train_data"]

        # This is a look up for performance measurements from inputname + configurationID
        input_config_map = (
            train_data[["inputname", "configurationID"] + performances]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )

        regret_map = input_config_map.groupby("inputname").transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        rank_map = input_config_map.groupby("inputname").transform(
            lambda x: stats.rankdata(x, method="min")
        )

        # Prepare and select training/test data according to random split
        train_input_mask = input_features.index.isin(train_inp)
        test_input_mask = input_features.index.isin(test_inp)

        train_config_mask = config_features.index.isin(train_cfg)
        test_config_mask = config_features.index.isin(test_cfg)

        input_preprocessor.fit(input_features[train_input_mask])
        config_preprocessor.fit(config_features[train_config_mask])

        input_arr = torch.from_numpy(
            input_preprocessor.transform(input_features)
        ).float()
        config_arr = torch.from_numpy(
            config_preprocessor.transform(config_features)
        ).float()

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
                epochs=epochs,
                performance=performances[0],
            )
            input_emb, config_emb = best_models
            input_emb = input_emb.cpu()
            config_emb = config_emb.cpu()

            with torch.no_grad():
                input_embeddings = input_emb(input_arr).numpy()
                config_embeddings = config_emb(config_arr).numpy()

        elif method == "tsne":
            # T-SNE from scikit-learn has no fit/transform split
            # Therefore we fit on the whole dataset, which is not direcly comparable
            # Also, we should consider different initializations to find the best embedding with TSNE
            # See: https://scikit-learn.org/stable/modules/manifold.html#t-sne
            input_emb = TSNE(dimensions)
            config_emb = TSNE(dimensions)

            input_embeddings = input_emb.fit_transform(input_arr)
            config_embeddings = config_emb.fit_transform(config_arr)
        elif method == "pca":
            input_emb = PCA(dimensions).fit(train_input_arr)
            config_emb = PCA(dimensions).fit(train_config_arr)

            input_embeddings = input_emb.transform(input_arr)
            config_embeddings = config_emb.transform(config_arr)
        elif method == "original":
            input_emb = None
            config_emb = None

            input_embeddings = input_arr
            config_embeddings = config_arr

        # Store model
        representation_file = os.path.join(
            result_dir, f"{method}_split{data_split['split']}.p"
        )
        torch.save(
            {
                "input_emb": input_emb,
                "config_emb": config_emb,
                "train_inp": train_inp,
                "train_cfg": train_cfg,
                "test_inp": test_inp,
                "test_cfg": test_cfg,
                "split": data_split["split"],
                "seed": random_seed,
                # "system": system,
            },
            open(representation_file, "wb"),
        )

        ### Evaluation

        ## Performance Prediction from embedded representation
        train_input_repr = input_embeddings[train_input_mask]
        test_input_repr = input_embeddings[test_input_mask]

        train_config_repr = config_embeddings[train_config_mask]
        test_config_repr = config_embeddings[test_config_mask]

        train_mape, test_mape = evaluate_prediction(
            performance_column=performances[0],
            data_split=data_split,
            train_input_repr=train_input_repr,
            train_config_repr=train_config_repr,
            test_input_repr=test_input_repr,
            test_config_repr=test_config_repr,
        )
        print(f"MAPE:{train_mape:.3f} / {test_mape:.3f}")
        mape.append((train_mape, test_mape))

        input_config_map_all = (
            perf_matrix[["inputname", "configurationID"] + performances]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )

        rank_arr_all = torch.from_numpy(
            input_config_map_all.groupby("inputname", as_index=False)
            .transform(lambda x: stats.rankdata(x, method="min"))
            .pivot_table(
                index="inputname", columns="configurationID", values=performances[0]
            )
            .values
        )
        regret_arr_all = torch.from_numpy(
            input_config_map_all.groupby("inputname", as_index=False)
            .transform(lambda x: ((x - x.min()) / (x.max() - x.min())))
            .pivot_table(
                index="inputname", columns="configurationID", values=performances[0]
            )
            .values
        )

        ## Detailed retrieval results
        result_df = evaluate_retrieval(
            topk_values=topk_values,
            topr_values=topr_values,
            rank_arr=rank_arr_all,
            regret_arr=regret_arr_all,
            train_input_mask=train_input_mask,
            test_input_mask=test_input_mask,
            train_config_mask=train_config_mask,
            test_config_mask=test_config_mask,
            input_embeddings=input_embeddings,
            config_embeddings=config_embeddings,
        )
        dfs.extend(result_df)

    # Aggregate MAPE results
    mape_df = pd.DataFrame(mape, columns=["train", "test"]).agg(["mean", "std"])
    mape_df.to_csv(os.path.join(result_dir, "mape.csv"))

    # Aggregate retrieval results
    full_df = pd.concat(dfs)
    full_df.groupby(["mode", "split", "metric", "k"]).mean()
    full_df.to_csv(os.path.join(result_dir, "full_df.csv"))

    make_latex_tables(full_df, result_dir=result_dir)


def main(
    data_dir,
    system,
    performance,
    input_properties,
    method,
    dimensions,
    epochs,
    output_dir,
):
    run_timestamp = (
        datetime.datetime.now().isoformat(timespec="minutes", sep="-").replace(":", "-")
    )
    result_dir = os.path.join(
        output_dir, f"{run_timestamp}_{system}_{method}_d{dimensions}"
    )
    os.makedirs(result_dir, exist_ok=True)

    assert method not in ("pca", "tsne") or 2 <= dimensions <= 3

    (
        perf_matrix,
        input_features,
        config_features,
        all_performances,
        input_preprocessor,
        config_preprocessor,
    ) = load_data(
        system=system, data_dir=data_dir, input_properties_type=input_properties
    )
    performances = all_performances[0:1] if performance is None else [performance]
    assert all(p in all_performances for p in performances)

    print(f"Loaded data for `{system}`")
    print(f"perf_matrix:{perf_matrix.shape}")
    print(f"input_features(before preprocessing):{input_features.shape}")
    print(f"config_features(before preprocessing):{config_features.shape}")

    evaluate_cv(
        method,
        dimensions,
        epochs,
        result_dir,
        perf_matrix,
        input_features,
        config_features,
        input_preprocessor,
        config_preprocessor,
        performances,
    )


if __name__ == "__main__":
    ## Load and prepare data
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("performance")
    parser.add_argument(
        "-ip",
        "--input-properties",
        default="tabular",
        choices=["tabular", "embeddings"],
    )
    parser.add_argument(
        "-m", "--method", default="embed", choices=["embed", "pca", "tsne", "original"]
    )
    parser.add_argument(
        "-d", "--dimensions", help="Embedding dimensions", default=32, type=int
    )
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("-o", "--output", default="../results/")

    # args = parser.parse_args(["poppler", "size", "-m=pca", "-d=3", "--data-dir", "data/", "--epochs=20"])
    args = parser.parse_args(
        ["gcc", "size", "-m=embed", "-d=3", "--data-dir", "data/", "--epochs=20"]
    )
    # args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        system=args.system,
        performance=args.performance,
        input_properties=args.input_properties,
        method=args.method,
        dimensions=args.dimensions,
        epochs=args.epochs,
        output_dir=args.output,
    )

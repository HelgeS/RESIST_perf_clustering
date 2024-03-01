import argparse
import copy
import datetime
import functools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (
    evaluate_ic,
    evaluate_ii,
    evaluate_prediction,
    evaluate_retrieval,
    load_data,
    make_latex_tables,
    pearson_rank_distance_matrix,
    split_data_cv,
)
from learning import predict, rankNet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from torch import nn

# Purpose of this file
# We learn a simple embedding of the input and configuration vectors
# Input_embed() Config_embed() → joint embedding space


def train_model_rank(
    train_input_arr,
    train_config_arr,
    input_arr,
    config_arr,
    ranks_train,
    ranks_test,
    emb_size,
    device,
    iterations,
    dropout=0.0,
    lr=0.0001,
    hidden_dim=64,
    do_eval=True,
    do_normalize=True,
    verbose=True,
    use_scheduler=False,
    max_subtract=False,
    ranknet_weight_by_diff=False,
):
    # iterations = 5000
    # emb_size = 16
    # device = device
    # dropout = 0.0
    # lr = 0.0003
    # hidden_dim = 64
    # do_eval = True
    # do_normalize = True

    ## Models
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
        optimizer, "min", verbose=verbose
    )

    # Baseline: Does CE loss between expected and predicted scores
    # lnloss = ListNetLoss()

    # seems to work much better, but slower
    # does a form of triplet/pair mining internally
    lnloss = functools.partial(
        rankNet,
        weight_by_diff=ranknet_weight_by_diff,
        weight_by_diff_powed=False,
        max_subtract=max_subtract,
    )

    # TODO Should we adjust the list ranking loss to consider min distances?
    # This could be part of the distance matrix,
    # i.e. something like the cumsum to enforce min distances between items

    # Early stopping
    best_loss = np.inf
    best_test_score = 0

    train_input_arr = train_input_arr.to(device)
    train_config_arr = train_config_arr.to(device)
    inp_cfg_ranks_pt = torch.from_numpy(ranks_train["inp_cfg"]).float().to(device)
    cfg_inp_ranks_pt = torch.from_numpy(ranks_train["cfg_inp"]).float().to(device)
    inp_inp_ranks_pt = torch.from_numpy(ranks_train["inp_inp"]).float().to(device)
    cfg_cfg_ranks_pt = torch.from_numpy(ranks_train["cfg_cfg"]).float().to(device)

    for iteration in range(iterations):
        # TODO Optionally, sample some inputs and configs if dataset too large
        # Check for x264
        input_emb.train()
        config_emb.train()

        inp_emb = input_emb(train_input_arr)
        cfg_emb = config_emb(train_config_arr)  # .detach()

        if do_normalize:
            inp_emb = F.normalize(inp_emb)
            cfg_emb = F.normalize(cfg_emb)

        loss = 0

        distmat_inp = torch.cdist(inp_emb, inp_emb)
        loss += lnloss(
            distmat_inp,
            inp_inp_ranks_pt,
        )
        distmat_cfg = torch.cdist(cfg_emb, cfg_emb)
        loss += lnloss(
            distmat_cfg,
            cfg_cfg_ranks_pt,
        )

        distmat = torch.cdist(inp_emb, cfg_emb)
        loss += lnloss(
            distmat,
            inp_cfg_ranks_pt,  # [input_indices, :][:, config_indices],
        )
        loss += lnloss(
            distmat.T,
            cfg_inp_ranks_pt,  # [config_indices, :][:, input_indices],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_scheduler:
            scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss.detach().item()

            if do_eval:
                train_scores = eval(
                    input_emb=input_emb,
                    config_emb=config_emb,
                    input_arr=train_input_arr,
                    config_arr=train_config_arr,
                    verbose=True,
                )
                # TODO We want to know the score for the test data only
                val_scores = eval(
                    input_emb=input_emb,
                    config_emb=config_emb,
                    input_arr=input_arr,
                    config_arr=config_arr,
                    verbose=True,
                )

                if all(c > 0.99 for c in train_scores):
                    break

    return best_loss, best_test_score


@torch.no_grad()
def eval(input_emb, config_emb, input_arr, config_arr, ranks, key="", verbose=False):
    input_emb.eval()
    config_emb.eval()

    scores = calc_scores(
        inp_emb=input_emb(input_arr),
        cfg_emb=config_emb(config_arr),
        ranks=ranks,
    )
    avg_score = np.mean(scores)

    if verbose:
        res = " | ".join([f"{s:.3f}" for s in scores])
        print(f"{key}\t\t{avg_score:.3f} | {res}")

    return scores


def calc_scores(inp_emb, cfg_emb, ranks, input_indices=None, config_indices=None):
    if input_indices is None:
        input_indices = np.arange(inp_emb.shape[0])

    if config_indices is None:
        config_indices = np.arange(cfg_emb.shape[0])
    correct_inp_inp = np.mean(
        stats.rankdata(torch.cdist(inp_emb, inp_emb).numpy(), axis=-1)
        <= ranks["inp_inp"][input_indices, :][:, input_indices]
    )
    correct_cfg_cfg = np.mean(
        stats.rankdata(torch.cdist(cfg_emb, cfg_emb).numpy(), axis=-1)
        <= ranks["cfg_cfg"][config_indices, :][:, config_indices]
    )
    dist_ic = torch.cdist(inp_emb, cfg_emb).numpy()
    correct_inp_cfg = np.mean(
        stats.rankdata(dist_ic, axis=-1)
        <= ranks["inp_cfg"][input_indices, :][:, config_indices]
    )
    correct_cfg_inp = np.mean(
        stats.rankdata(dist_ic.T, axis=-1)
        <= ranks["cfg_inp"][config_indices, :][:, input_indices]
    )
    return (correct_inp_inp, correct_cfg_cfg, correct_inp_cfg, correct_cfg_inp)


def train_model(
    train_inp,
    train_cfg,
    train_input_arr,
    train_config_arr,
    regret_map,
    performance,
    emb_size,
    epochs,
    lr=0.003,
    hidden_dim=32,
    dropout=0.0,
):
    num_input_features = train_input_arr.shape[1]
    num_config_features = train_config_arr.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # TODO Implement performance prediction from latent space
    # perf_predict = nn.Sequential(
    #     nn.Linear(2 * emb_size, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 1),  # TODO Check with |P| outputs
    # )

    optimizer = torch.optim.AdamW(
        list(input_emb.parameters()) + list(config_emb.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )

    # lnloss = ListNetLoss()
    lnloss = functools.partial(
        rankNet,
        weight_by_diff=True,
        weight_by_diff_powed=False,
    )

    # Early stopping
    best_loss = np.inf
    best_loss_iter = 0
    best_logmsg = None
    best_models = (copy.deepcopy(input_emb), copy.deepcopy(config_emb))
    patience = 1000

    # TODO For our dataset size it is relatively cheap to calculate the embeddings for all inputs and configs.
    # We can every few iterations update a full collection and collect the hardest triplets from it.
    #
    # with torch.no_grad():
    #     emb_lookup = torch.empty(
    #         (train_input_arr.shape[0] + train_config_arr.shape[0], emb_size)
    #     )
    #     emb_lookup[: train_input_arr.shape[0]] = input_emb(train_input_arr)
    #     emb_lookup[train_input_arr.shape[0] :] = config_emb(train_config_arr)

    regret_arr = torch.from_numpy(
        regret_map.loc[(train_inp, train_cfg), :]
        .reset_index()
        .pivot_table(index="inputname", columns="configurationID", values=performance)
        .values
    ).to(device)

    inp_cfg_ranks = regret_arr.argsort(dim=-1).float()
    inp_cfg_ranks = inp_cfg_ranks / inp_cfg_ranks.max(dim=-1, keepdim=True).values

    cfg_inp_ranks = regret_arr.T.argsort(dim=-1).float()
    cfg_inp_ranks = cfg_inp_ranks / cfg_inp_ranks.max(dim=-1, keepdim=True).values

    inp_inp_ranks = torch.from_numpy(
        pearson_rank_distance_matrix(
            np.expand_dims(
                regret_map.loc[(train_inp, train_cfg), :]
                .reset_index()
                .pivot_table(
                    index="inputname", columns="configurationID", values=performance
                )
                .values,
                axis=-1,
            )
        ).squeeze(-1)
    ).to(device)
    cfg_cfg_ranks = torch.from_numpy(
        pearson_rank_distance_matrix(
            np.expand_dims(
                regret_map.loc[(train_inp, train_cfg), :]
                .reset_index()
                .pivot_table(
                    index="configurationID", columns="inputname", values=performance
                )
                .values,
                axis=-1,
            )
        ).squeeze(-1)
    ).to(device)

    train_input_arr = train_input_arr.to(device)
    train_config_arr = train_config_arr.to(device)

    total_loss = 0

    for iteration in range(epochs):
        optimizer.zero_grad()

        # batch, input_indices, config_indices = make_batch(
        #     train_inp,
        #     train_cfg,
        #     batch_size,
        #     input_map=input_map,
        #     config_map=config_map,
        #     rank_map=rank_map,
        # )
        # batch = batch.reshape((-1, 2)).to(device)
        # input_indices = input_indices.to(device)
        # config_indices = config_indices.to(device)
        # input_row = batch[:, 0] == 1
        # assert (
        #     batch.shape[1] == 2
        # ), "Make sure to reshape batch to two columns (type, index)"

        # embeddings = torch.empty((batch.shape[0], emb_size), device=device)
        # embeddings[input_row] = input_emb(train_input_arr[batch[input_row, 1]])
        # embeddings[~input_row] = config_emb(train_config_arr[batch[~input_row, 1]])

        # embeddings = F.normalize(embeddings, dim=1)

        # loss = nn.functional.triplet_margin_loss(
        #     anchor=embeddings[0::3],
        #     positive=embeddings[1::3],
        #     negative=embeddings[2::3],
        # )

        # # TODO Add alternative ranking losses that prioritize top ranks
        # # TODO Make lnloss weight a parameter + hpsearch
        # distmat = torch.cdist(embeddings[input_row], embeddings[~input_row])
        # loss += 0.05 * lnloss(
        #     torch.argsort(distmat, dim=1).float(),
        #     rank_arr[input_indices, :][:, config_indices].float(),
        # )
        # loss += 0.05 * lnloss(
        #     torch.argsort(distmat.T, dim=1).float(),
        #     rank_arr_cfg[config_indices, :][:, input_indices].float(),
        # )

        ## Here we take some inputs + configs and apply listnet ranking loss
        input_indices = torch.randint(
            train_input_arr.shape[0], size=(8,), device=device
        )
        config_indices = torch.randint(
            train_config_arr.shape[0], size=(8,), device=device
        )

        input_embeddings = predict(input_emb, train_input_arr[input_indices])
        config_embeddings = predict(config_emb, train_config_arr[config_indices])

        # distmat = torch.cdist(input_embeddings, config_embeddings)
        # loss = lnloss(
        #     distmat,
        #     inp_cfg_ranks[input_indices, :][:, config_indices],
        # )
        # loss += lnloss(
        #     distmat.T,
        #     cfg_inp_ranks[config_indices, :][:, input_indices],
        # )
        loss = 0
        distmat_ii = torch.cdist(input_embeddings, input_embeddings)
        loss += lnloss(
            distmat_ii,
            inp_inp_ranks[input_indices, :][:, input_indices],
        )

        distmat_cc = torch.cdist(config_embeddings, config_embeddings)
        # loss += lnloss(
        #     distmat_cc,
        #     cfg_cfg_ranks[config_indices, :][:, config_indices],
        # )

        ## Here we take the distance matrix and sample easy positive/hard negative
        # Rank mismatch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()

        if iteration > 0 and iteration % 50 == 0:
            total_loss /= 10
            scheduler.step(total_loss)

            with torch.no_grad():
                inputembs = predict(input_emb, train_input_arr)

                if inputembs.std(axis=0).mean() < 0.05:
                    print("Input embeddings collapsed")
                    break

                (
                    icc_best_rank,
                    icc_avg_rank,
                    icc_best_regret,
                    icc_avg_regret,
                ) = evaluate_ic(
                    input_representation=inputembs,
                    config_representation=config_emb(train_config_arr),
                    regret_arr=regret_arr,
                    k=5,
                )
                iii_ranks, iii_regret, _ = evaluate_ii(
                    input_representation=inputembs,
                    regret_arr=regret_arr,
                    n_neighbors=5,
                    n_recs=[1, 3, 5, 15, 25],
                )

                cc_sort = torch.mean(
                    (
                        distmat_cc.argsort(dim=-1)
                        == cfg_cfg_ranks[config_indices, :][:, config_indices].argsort(
                            dim=-1
                        )
                    ).float()
                )
                ii_sort = torch.mean(
                    (
                        distmat_ii.argsort(dim=-1)
                        == inp_inp_ranks[input_indices, :][:, input_indices].argsort(
                            dim=-1
                        )
                    ).float()
                )

                print(cc_sort, ii_sort)
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

    torch.manual_seed(random_seed)
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

        # TODO: here we rank the inputs per configuration
        # To make the input performances comparable, we must normalize them
        # or use the regret? or use the previously calculated ranks?

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
                regret_map=regret_map,
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

            # We set the perplexity to be smaller than the number of samples
            # or the default value
            # TODO This should be optimized per dataset, ideally automatically
            input_emb = TSNE(dimensions, perplexity=min(30, input_arr.shape[0] - 1))
            config_emb = TSNE(dimensions, perplexity=min(30, config_arr.shape[0] - 1))

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

        regret_arr_all = torch.from_numpy(
            perf_matrix[["inputname", "configurationID"] + performances]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
            .groupby("inputname", as_index=False)
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
        ["poppler", "size", "-m=embed", "-d=8", "--data-dir", "data/", "--epochs=10000"]
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

import itertools
import numpy as np
import torch

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

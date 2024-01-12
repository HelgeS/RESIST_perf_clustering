import math
import itertools
import pickle

from scipy import stats
from joblib import Parallel, delayed
import pandas as pd

from common import load_x264

perf_matrix, input_features, config_features, performances = load_x264(data_dir="data/")

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

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)

## This implements a cache for the correlation calculation
# TODO We can precalculate it for the whole dataset and then index appropriately


# Inputs
def calc_corr_inp(inp1, inp2):
    return tuple(sorted((inp1, inp2))), rank_map.loc[inp1].corrwith(rank_map.loc[inp2])


inpcombs = math.comb(len(all_input_names), 2)
print(f"Inputs ({inpcombs})")
inpcorrs = {}
for k, v in Parallel(n_jobs=-1, verbose=10)(
    delayed(calc_corr_inp)(inp1, inp2)
    for inp1, inp2 in itertools.combinations(all_input_names, 2)
):
    inpcorrs[k] = v


# Configurations
def calc_corr_cfg(cfg1, cfg2):
    return tuple(sorted((cfg1, cfg2))), rank_map.xs(cfg1, level=1).corrwith(
        rank_map.xs(cfg2, level=1)
    )


cfgcombs = math.comb(len(all_config_ids), 2)
print(f"Configurations ({cfgcombs})")
cfgcorrs = {}
for k, v in Parallel(n_jobs=-1, verbose=10)(
    delayed(calc_corr_cfg)(cfg1, cfg2)
    for cfg1, cfg2 in itertools.combinations(all_config_ids, 2)
):
    cfgcorrs[k] = v

pickle.dump(
    {
        "dataset": "x264",
        "input_correlations": inpcorrs,
        "config_correlations": cfgcorrs,
    },
    open("data/x264_correlations.p", "wb"),
)

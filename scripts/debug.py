import sys
import random

import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

np.set_printoptions(edgeitems=30, linewidth=1000,
                    formatter=dict(float=lambda x: "%.3g" % x))


def preprocess_function(df):
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["amount"].pct_change()
    df.dropna(inplace=True)
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    return df


if __name__ == "__main__":
    sys.path.append("build")
    import trade_env
    # num_feature, bs, max_timestep
    bs = 128
    vec_trade = trade_env.VecTrade(5, bs, 48 * 2, 0.05)
    instruments = glob("data/*.csv.gz")
    ins = random.choice(instruments)
    df = preprocess_function(pd.read_csv(
        ins, parse_dates=["time"], index_col="time"))

    code = df["code"][0]
    raw_names = ['open', 'high', 'low', 'close', 'volume']
    fea_names = ['feature_open', 'feature_high',
                 'feature_low', 'feature_close', 'feature_volume']

    vec_trade.Load(code, df[raw_names].values, df[fea_names].values)

    obs = vec_trade.Reset()
    pbar = tqdm(total=1000 * 1000)
    while pbar.n < pbar.total:
        a = [random.choice([0, 1]) for _ in range(bs)]
        obs, reward, done = vec_trade.Step(a)
        pbar.update(bs)
    # print(i, a, reward, done, obs[0][-1])

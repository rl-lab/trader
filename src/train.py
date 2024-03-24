from gym_trading_env.renderer import Renderer
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from gym_trading_env.environments import TradingEnv, MultiDatasetTradingEnv
import gymnasium as gym


def preprocess_function(df):
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["amount"] / df["amount"]
    df.dropna(inplace=True)
    return df


num_stocks = 10
dfs = [pd.read_csv(each, parse_dates=["time"], index_col="time")
       for each in tqdm(glob("data/*.csv.gz")[:num_stocks], desc="reading datasets")]


def reward_function(history):
    # log (p_t / p_t-1 )
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


def make_env():
    env = MultiDatasetTradingEnv(
        name="hsi300",
        datasets=dfs,
        windows=1,
        positions=[-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        initial_position='random',
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        preprocess=preprocess_function,
        reward_function=reward_function,
        portfolio_initial_value=1000,
        max_episode_duration=200,
        verbose=0,
    )

    env.add_metric('Position Changes', lambda history: np.sum(
        np.diff(history['position']) != 0))
    env.add_metric('Episode Lenght', lambda history: len(history['position']))
    return env


num_envs = 16
env = gym.vector.SyncVectorEnv([make_env] * num_envs)

observation, info = env.reset()
print(observation.shape)
# print(info["data_code"], info["date"], info["data_open"])


pbar = tqdm(total=1000*1000)

while True:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    pbar.update(num_envs)


# while not done and not truncated:
#     action = env.action_space.sample()
#     # action = policy_net(observation)
#     observation, reward, done, truncated, info = env.step(action)
#     # data.append([observation, reward])
#
#
# env.save_for_render()
# renderer = Renderer(render_logs_dir="render_logs")
# renderer.run()

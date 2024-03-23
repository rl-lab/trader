import pandas as pd
import numpy as np
import time
from gym_trading_env.environments import TradingEnv
import gymnasium as gym


# Import your datas
df = pd.read_csv("BTC_USD-Hourly.csv.gz",
                 parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# WARNING : the column names need to contain keyword 'feature' !
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()
df.dropna(inplace=True)


# Create your own reward function with the history object
def reward_function(history):
    # log (p_t / p_t-1 )
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


# env = gym.make(
#     "TradingEnv",
env = TradingEnv(
    name="BTCUSD",
    df=df,
    windows=5,
    # From -1 (=SHORT), to +1 (=LONG)
    positions=[-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2],
    initial_position='random',  # Initial position
    trading_fees=0.01/100,  # 0.01% per stock buy / sell
    borrow_interest_rate=0.0003/100,  # per timestep (= 1h here)
    reward_function=reward_function,
    portfolio_initial_value=1000,  # in FIAT (here, USD)
    max_episode_duration=500,
)

env.add_metric('Position Changes', lambda history: np.sum(
    np.diff(history['position']) != 0))
env.add_metric('Episode Lenght', lambda history: len(history['position']))

done, truncated = False, False
observation, info = env.reset()
print(info["date"], "btc price", info["data_open"])

while not done and not truncated:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)

env.save_for_render()

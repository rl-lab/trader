import pandas as pd
import numpy as np
import time
from gym_trading_env.environments import TradingEnv
import gymnasium as gym


# Import your datas
df = pd.read_csv("BTC_USD-Hourly.csv.gz", parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

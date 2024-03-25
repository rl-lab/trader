from gym_trading_env.renderer import Renderer
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from gym_trading_env.environments import TradingEnv, MultiDatasetTradingEnv
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim


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
dfs = [preprocess_function(pd.read_csv(each, parse_dates=["time"], index_col="time"))
       for each in tqdm(glob("data/*.csv.gz")[:num_stocks], desc="reading datasets")]


def reward_function(history):
    # log (p_t / p_t-1 )
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


positions = [-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2]


def make_env():
    env = MultiDatasetTradingEnv(
        name="hsi300",
        datasets=dfs,
        windows=1,
        positions=positions,
        initial_position='random',
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
        portfolio_initial_value=1000,
        max_episode_duration=200,
        verbose=0,
    )

    env.add_metric('Position Changes', lambda history: np.sum(
        np.diff(history['position']) != 0))
    env.add_metric('Episode Lenght', lambda history: len(history['position']))
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(5, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(
            nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


num_envs = 16
device = torch.device("cuda" if torch.cuda.is_available()
                      and args.cuda else "cpu")
env = gym.vector.SyncVectorEnv([make_env] * num_envs)
agent = Agent(num_actions=len(positions)).to(device)
optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

observation, info = env.reset()
print(observation.shape)

pbar = tqdm(total=1000*1000)
while True:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    pbar.update(num_envs)


#
# env.save_for_render()
# renderer = Renderer(render_logs_dir="render_logs")
# renderer.run()

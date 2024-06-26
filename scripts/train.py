import sys
import time
import random

import pickle
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import talib
import wandb

np.set_printoptions(edgeitems=30, linewidth=1000,
                    formatter=dict(float=lambda x: "%.3g" % x))
pd.set_option('display.float_format', '{:.2f}'.format)

use_wandb = True
if use_wandb:
    wandb.login(key="585ae2121002eef020cd686fede2bce79a15faf3")
    wandb.init(project="trader")

fea_names = ["feature_close", "feature_open",
             "feature_high", "feature_low", "feature_volume",
             "SMA_50", "SMA_200", "BB_UPPER", "BB_MIDDLE", "BB_LOWER",
             "RSI_14", "MACD", "MACD_SIGNAL", "EMA_20", "ATR_14",
             "CCI_20", "Aroon_Osc"]


def preprocess_function(df, seperate="2022-12-31"):
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    given_date = pd.to_datetime('2023-01-01')
    df = df.astype('double')

    # normalize price, not sure whether to do this
    df["close"] /= df["close"][0]
    df["open"] /= df["open"][0]
    df["high"] /= df["high"][0]
    df["low"] /= df["low"][0]

    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["amount"].pct_change()

    df['SMA_50'] = talib.SMA(df['close'].values, timeperiod=50)
    df['SMA_200'] = talib.SMA(df['close'].values, timeperiod=200)

    upper_band, middle_band, lower_band = talib.BBANDS(
        df['close'].values, timeperiod=20)
    df['BB_UPPER'] = upper_band
    df['BB_MIDDLE'] = middle_band
    df['BB_LOWER'] = lower_band

    df['RSI_14'] = talib.RSI(df['close'].values, timeperiod=14)

    macd, signal, _ = talib.MACD(
        df['close'].values, fastperiod=8, slowperiod=21, signalperiod=9)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = signal

    df['EMA_20'] = talib.EMA(df['close'].values, timeperiod=20)
    df['ATR_14'] = talib.ATR(
        df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    df['CCI_20'] = talib.CCI(
        df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
    df['Aroon_Osc'] = talib.AROONOSC(
        df['high'].values, df['low'].values, timeperiod=14)

    df.dropna(inplace=True)
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    df = df.astype('float')
    return df[df.index < given_date], df[df.index >= given_date]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions, num_features, dim=128):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(num_features, dim)),
            nn.ReLU(),
            layer_init(nn.Linear(dim, dim)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(dim, dim)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(
            nn.Linear(dim, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(dim, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

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


if __name__ == "__main__":
    sys.path.append("build")
    import trade_env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 128
    T = 256
    MAX_TRADE_STEPS = 12 * 4 * 2  # two days
    AMT_EACH_STEP = 0.05  # buy this fixed position each step

    raw_names = ['open', 'high', 'low', 'close', 'volume']

    # create a custom data env with batch operation
    train_env = trade_env.VecTrade(
        len(fea_names), num_envs, MAX_TRADE_STEPS, AMT_EACH_STEP)
    test_env = trade_env.VecTrade(
        len(fea_names), num_envs, MAX_TRADE_STEPS, AMT_EACH_STEP)

    instruments = glob("data/*.csv.gz")
    for i, ins in enumerate(tqdm(instruments)):
        df = pd.read_csv(ins, parse_dates=["time"], index_col="time")
        code = df["code"][0]
        df.drop("code", axis=1, inplace=True)
        traindf, testdf = preprocess_function(df, seperate="2022-12-31")

        if i == 0:
            print("example data")
            print(traindf[:10].transpose())

        # load the pandas df data into CPP vector<float>, for fast operations
        if len(traindf) > MAX_TRADE_STEPS * 10:
            train_env.Load(
                code, traindf[raw_names].values, traindf[fea_names].values)
        if len(testdf) > MAX_TRADE_STEPS * 10:
            test_env.Load(code, testdf[raw_names].values,
                          testdf[fea_names].values)

    # num_features = len(fea_names) + 2, because env always append a
    # position and a time-left feature
    agent = Agent(num_actions=2, num_features=len(fea_names) + 2).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    # train part
    next_obs = train_env.Reset()
    next_obs = torch.Tensor(next_obs).to(device)

    obs = torch.zeros((T, *next_obs.shape)).to(device)
    actions = torch.zeros((T, num_envs)).to(device)
    logprobs = torch.zeros((T, num_envs)).to(device)
    rewards = torch.zeros((T, num_envs)).to(device)
    dones = torch.zeros((T, num_envs)).to(device)
    values = torch.zeros((T, num_envs)).to(device)

    next_done = torch.zeros(num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_envs,
                    agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_envs,
                    agent.lstm.hidden_size).to(device),
    )

    test_next_obs = test_env.Reset()
    test_next_obs = torch.Tensor(test_next_obs).to(device)
    test_next_done = torch.zeros(num_envs).to(device)
    test_next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_envs,
                    agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_envs,
                    agent.lstm.hidden_size).to(device),
    )

    ma_rewards = deque(maxlen=100)
    test_rewards = deque(maxlen=100)
    last_test_time = 0
    ma_epilen = deque(maxlen=100)
    pbar = tqdm(total=100*1000*1000)

    last_wandb_time = time.time()
    traj_counter = np.zeros(num_envs)

    for learn_step in range(1000 * 1000):
        initial_lstm_state = (
            next_lstm_state[0].clone(), next_lstm_state[1].clone())

        epi_rewards = []
        epi_len = []
        for step in range(0, T):
            obs[step] = next_obs
            dones[step] = next_done
            # with open('arrays.pkl', 'wb') as f:
            #     pickle.dump((next_obs.cpu().numpy(), next_lstm_state[0].cpu().numpy(), next_lstm_state[1].cpu().numpy()), f)

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, next_done = train_env.Step(
                action.cpu().numpy().tolist())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            epi_rewards += reward[next_done == 1].tolist()
            epi_len += traj_counter[next_done == 1].tolist()
            traj_counter = (traj_counter + 1) * (1 - next_done)

            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(next_done).to(device)

        ma_rewards.append(np.mean(epi_rewards))
        ma_epilen.append(np.mean(epi_len))
        pbar.set_description(
            f"train reward: {np.mean(ma_rewards)}, test reward: {np.mean(test_rewards)}")
        if use_wandb:
            wandb.log({
                "reward": np.mean(ma_rewards),
                "trajlen": np.mean(ma_epilen),
            })
        pbar.update(num_envs * T)

        # bootstrap value if not done
        Gamma = 0.99
        Lambda = 0.95
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + Gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + Gamma * \
                    Lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, len(fea_names) + 2))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network

        # Optimizing the policy and value network
        num_minibatches = 8
        update_epochs = 2
        envsperbatch = num_envs // num_minibatches
        envinds = np.arange(num_envs)
        flatinds = np.arange(T * num_envs).reshape(T, num_envs)
        clipfracs = []

        for epoch in range(update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                # be really careful about the index
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds],
                     initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   0.2).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if True:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -0.2,
                        0.2,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - 0.001 * entropy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        if use_wandb:
            wandb.log({
                "pg_loss": pg_loss.item(),
                "v_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "explained_variance": explained_var,
            })

        if time.time() - last_test_time > 300:
            # test every 5 min, T * 100 approx use 40 seconds on V100
            agent.eval()

            last_test_time = time.time()
            epi_rewards = []
            epi_len = []
            for step in tqdm(range(0, T * 100)):
                with torch.no_grad():
                    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                        test_next_obs, test_next_lstm_state, test_next_done)
                test_next_obs, reward, test_next_done = train_env.Step(
                    action.cpu().numpy().tolist())
                epi_rewards += reward[test_next_done == 1].tolist()

                test_next_obs, test_next_done = torch.Tensor(test_next_obs).to(
                    device), torch.Tensor(test_next_done).to(device)

            test_rewards.append(np.mean(epi_rewards))
            if use_wandb:
                wandb.log({
                    "test_reward": np.mean(test_rewards),
                })
            agent.train()


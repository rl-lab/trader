from gym_trading_env.renderer import Renderer
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from torch.distributions.categorical import Categorical
from gym_trading_env.environments import TradingEnv, MultiDatasetTradingEnv
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import wandb


def preprocess_function(df):
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["amount"].pct_change()
    df.dropna(inplace=True)
    return df


num_stocks = 10
dfs = [preprocess_function(pd.read_csv(each, parse_dates=["time"], index_col="time"))
       for each in tqdm(glob("data/*.csv.gz")[:num_stocks], desc="reading datasets")]


def reward_function(history):
    # log (p_t / p_t-1 )
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


positions = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2])


def make_env():
    env = MultiDatasetTradingEnv(
        name="hsi300",
        datasets=dfs,
        windows=None,
        positions=positions.tolist(),
        initial_position='random',
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
        portfolio_initial_value=1000,
        max_episode_duration=400,
        verbose=0,
    )

    env.add_metric(
        'avg_position', lambda history: np.mean(history['position']))
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(7, 128)),
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


if __name__ == "__main__":
    num_envs = 16
    T = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.vector.SyncVectorEnv([make_env] * num_envs)
    agent = Agent(num_actions=len(positions)).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    next_obs, info = env.reset()
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
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    ma_rewards = deque(maxlen=100)
    ma_positions = deque(maxlen=100)
    pbar = tqdm(total=100*1000*1000)

    # wandb.login(key="585ae2121002eef020cd686fede2bce79a15faf3")
    # wandb.init(project="trader")

    last_wandb_time = time.time()
    while True:
        initial_lstm_state = (
            next_lstm_state[0].clone(), next_lstm_state[1].clone())

        for step in range(0, T):
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            ma_positions.append(positions[action.cpu().numpy()].mean())

            next_obs, reward, terminations, truncations, infos = env.step(
                action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(next_done).to(device)
            pbar.update(num_envs)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    ma_rewards.append(info['reward'])
                    pbar.set_description(
                        f"reward: {np.mean(ma_rewards)} positions: {np.mean(ma_positions)}")
                if time.time() - last_wandb_time > 5:
                    last_wandb_time = time.time()
                    wandb.log({
                        "reward": np.mean(ma_rewards),
                        "positions": np.mean(ma_positions)
                    })

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
        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network

        # Optimizing the policy and value network
        num_minibatches = 8
        update_epochs = 4
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
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
                loss = pg_loss - 0.01 * entropy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y


# env.save_for_render()
# renderer = Renderer(render_logs_dir="render_logs")
# renderer.run()

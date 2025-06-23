import enum
import os
import time
import gymnasium as gym
import math
import random
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Envs(enum.Enum):
    ICELAKE = 0
    CARTPOLE = 1


class DQN(nn.Module):
    """
    Network architecture for Deep Q-Learning
    """

    def __init__(self, input_size: int, observation_size: int):
        super(DQN, self).__init__()

        # Architecture of the CNN
        self.arch: nn.Sequential = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

        # # HE initialization
        # for layer in [self.arch]:
        #     for module in layer:
        #         if isinstance(module, nn.Linear):
        #             nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, x):
        return self.arch(x)


Experience = namedtuple("Experience", ("st", "at", "rt", "st1"))
Experience.__doc__ = """Experience includes: 
Current State
Current Action
Current Reward
Next State"""


class ReplayBuffer(object):
    """
    Replay Buffer to hold experience
    """

    def __init__(self, max_size: int):
        self.memory: deque[Experience] = deque([], maxlen=max_size)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, sample_size: int):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class EpsilonGreedyPolicy(object):
    """
    Epsilon Greedy policy for choosing actions
    """

    def __init__(
        self, device: torch.device, lower: float, upper: float, decay_rate: float
    ):
        self.E_LOWER = lower
        self.E_UPPER = upper
        self.E_DECAY = decay_rate
        self.device = device

    def choose_action(
        self,
        episode: int,
        state: torch.Tensor,
        device: torch.device,
        env: gym.Env,
        network: nn.Module,
        dtype: torch.dtype,
        environment: Envs = Envs.ICELAKE,
    ):
        threshold = max(self.E_LOWER, self.E_UPPER * (self.E_DECAY**episode))
        print(threshold)

        if random.random() <= threshold:
            return torch.tensor(
                [[env.action_space.sample()]], dtype=torch.long, device=device
            )

        with torch.no_grad():
            # Get the action with the largest expected reward
            e_values = network(state)
            if environment == Envs.ICELAKE:
                return torch.argmax(e_values).view(1, 1)
            if environment == Envs.CARTPOLE:
                return torch.max(e_values, 1).indices.view(1, 1)


def preprocess_state(
    state,
    num_states: int,
    device: torch.device,
    dtype: torch.dtype,
    environment: Envs = Envs.ICELAKE,
):
    if environment == Envs.ICELAKE:
        state_t = torch.zeros(num_states, dtype=dtype, device=device)
        state_t[state] = 1
        return state_t
    if environment == Envs.CARTPOLE:
        return torch.tensor(state, dtype=dtype, device=device).unsqueeze(0)


def batch_learn(
    gamma: float,
    target: nn.Module,
    policy: nn.Module,
    experiences: list[Experience],
    optimizer: optim.AdamW,
    device,
    divergence: list | None = None,
    environment: Envs = Envs.ICELAKE,
):
    # Transpose the batch
    batch = Experience(*zip(*experiences))
    # print(batch)
    # Turn the parameters into tensor batches
    st_batch = torch.tensor([])
    if environment == Envs.ICELAKE:
        st_batch = torch.stack(batch.st)
    if environment == Envs.CARTPOLE:
        st_batch = torch.cat(batch.st)
    at_batch = torch.cat(batch.at).long()
    rt_batch = torch.cat(batch.rt)

    # print("AT: ", at_batch.shape)
    # print("ST: ", st_batch.shape)
    # print("RT: ", rt_batch.shape)
    # exit()
    # Calculate state action values
    state_action_values = policy(st_batch).gather(1, at_batch)

    # Create a filter for non-terminal states
    non_terminal_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.st1)), dtype=torch.bool, device=device
    )
    # print("Nt_mask: ", non_terminal_mask.shape)
    # Create a tensor which only includes non_terminal next states
    if environment == Envs.CARTPOLE:
        non_terminal_next = torch.cat([st1 for st1 in batch.st1 if st1 is not None])
    else:
        non_terminal_next = torch.stack([st1 for st1 in batch.st1 if st1 is not None])
    # print("Nt_next: ", non_terminal_next.shape)

    # Apply the filter to separate states which need their expected value calculated
    filtered_expected = torch.zeros(len(experiences), device=device)
    # Extract the max value along the input dimension
    with torch.no_grad():
        temp_expect = target(non_terminal_next)
        # print("temp_exp", temp_expect.shape)
        filtered_expected[non_terminal_mask] = torch.max(temp_expect, 1).values
    # print("filter_exp: ", filtered_expected.shape)
    # print(filtered_expected)

    # Create the target values
    # r_i
    #   for terminal next_state
    # r_i + gamma * max_a' Q(current_state, a'; parameters)
    #   for non-terminal next_state
    target_values = rt_batch + (gamma * filtered_expected)
    # print("t_vals", target_values.shape)
    # print(target_values)

    # Track divergence if given a divergence list
    if divergence is not None:
        diff = target_values - state_action_values
        mse = torch.mean(diff**2).item()
        abs_diff = torch.mean(torch.abs(diff)).item()
        max_diff = torch.max(torch.abs(diff)).item()
        # divergence.append((mse, abs_diff, max_diff))
        divergence.append(mse)

    # Compute loss
    criterion = nn.SmoothL1Loss()
    # print(f"action: {state_action_values.shape}, target: {target_values.unsqueeze(1).shape}")
    loss = criterion(state_action_values, target_values.unsqueeze(1))
    # print("loss", loss.shape)
    # print(loss)

    # Perform Optimization
    optimizer.zero_grad()
    loss.backward()
    # Clip the gradient
    nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()
    # exit()


def perform_DQN_episodes(
    state_size: int,
    episodes: int,
    batch_size: int,
    gamma: float,
    tau: float,
    env: gym.Env,
    target: nn.Module,
    policy: nn.Module,
    egreedy: EpsilonGreedyPolicy,
    replay_buffer: ReplayBuffer,
    optimizer: optim.AdamW,
    device: torch.device,
    dtype: torch.dtype,
    plot: bool = False,
    label: str = "",
    environment: Envs = Envs.ICELAKE,
):
    total_steps = 0
    rewards = []
    divergence = [] if plot else None
    for e in range(episodes):
        state, info = env.reset()
        state = preprocess_state(
            state, state_size, device, dtype, environment=environment
        )
        # Repeat until a terminal state is reached
        for t in count():
            total_steps += 1
            # Take an action according to the epsilon greedy policy policy
            action: torch.Tensor = egreedy.choose_action(
                e if environment == Envs.ICELAKE else total_steps,
                state,
                device,
                env,
                policy,
                dtype,
                environment=environment,
            )
            observation, reward, terminated, truncated, _ = env.step(action.item())

            # Create tensors out of the current environment
            reward = torch.tensor([reward], device=device)
            next_state = (
                None
                if terminated
                else preprocess_state(
                    observation, state_size, device, dtype, environment=environment
                )
            )

            # Add the experience to the replay buffer
            replay_buffer.push(state, action, reward, next_state)

            # use batch learning if the replay buffer is large enough
            if len(replay_buffer) > batch_size:
                batch_learn(
                    gamma,
                    target,
                    policy,
                    replay_buffer.sample(batch_size),
                    optimizer,
                    device,
                    divergence=divergence,
                    environment=environment,
                )

            # Soft update
            target_state_dict = target.state_dict()
            policy_state_dict = policy.state_dict()
            for key in policy_state_dict:
                target_state_dict[key] = policy_state_dict[
                    key
                ] * tau + target_state_dict[key] * (1 - tau)
            target.load_state_dict(target_state_dict)

            # If the environment hasn't terminated, advance state; otherwise terminate episode
            if not (terminated or truncated) and next_state is not None:
                # print(next_state)
                state = next_state
            else:
                # print(
                # f"{'truncated' if truncated else ''} {'terminated' if terminated else ''}"
                # )
                # Add reward to rewards
                if environment == Envs.CARTPOLE:
                    rewards.append(t)
                else:
                    rewards.append(reward)
                break

        # If a plot was given, visualize all episodes every 10 episodes
        if plot and e % 10 == 9:
            visualize_episodes(None, rewards, dtype, fig_num=1)

    # If a plot was given, visualize all rewards
    if plot:
        visualize_episodes(
            os.path.join(label, "divergence.png"), divergence, dtype, fig_num=2
        )
        visualize_episodes(os.path.join(label, "training_rewards.png"), rewards, dtype)


def visualize_episodes(
    label: str | None, rewards, dtype: torch.dtype, fig_num: int = 1
):
    show = label is not None
    plt.figure(fig_num)
    rewards_t = torch.tensor(rewards, dtype=dtype)
    plt.clf()
    if show:
        plt.title("Finished")
    else:
        plt.title("Training...")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(rewards_t.numpy())

    # Plot 100 episode averages
    if len(rewards_t) > 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        if show:
            plt.title(f"Finished, best 100 avg: {means.numpy().max()}")

    if plt.isinteractive():
        plt.pause(0.001)

    if show:
        plt.savefig(label)

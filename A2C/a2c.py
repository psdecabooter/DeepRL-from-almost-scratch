from collections import deque
from enum import Enum
import math
from queue import Queue
import random
from typing import NamedTuple
import gymnasium as gym
from matplotlib import pyplot as plt
from sympy import Transpose
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.distributions import Categorical


class A2C(nn.Module):
    def __init__(
        self,
        action_size: int,
        network_size: int,
        shared_network: nn.Module,
    ) -> None:
        super(A2C, self).__init__()

        # Shared Parameters
        self.main = shared_network

        self.policy = nn.Linear(network_size, action_size)
        self.value = nn.Linear(network_size, 1)

    def forward(self, state):
        """
        Feed Forward function for the A2C module

        Args:
            state (tensor): input tensor to the A2C Network

        Returns:
            (tensor,tensor): the output tensor for the policy network and the value network
        """
        # Shared parameters for the policy and value network
        state = self.main(state)
        # print(state.shape)
        # (policy network, value network)
        return F.softmax(self.policy(state), dim=-1), self.value(state)


class Experience(NamedTuple):
    """
    Experience includes:
        Current State
        Current Action
        Current Reward
        Next State
    """

    state: torch.Tensor
    action: torch.Tensor
    log_probability: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    done: bool


class ENVS(Enum):
    cartpole = 0
    pong = 1


class A2C_algorithm:
    def __init__(
        self,
        num_environments: int,
        shared_network: nn.Module,
        env: gym.vector.VectorEnv,
        device: torch.device,
        n_act: int,
        beta: float,
        gamma: float,
        learning_rate: float,
        value_coef: float,
        tmax: int,
        clip_norm: float,
        network_size: int,
        environment_type: ENVS = ENVS.cartpole,
        preprocess_fn=None,
    ):
        # Hyperparameters
        self.NENVS = num_environments
        self.CLIP_NORM = clip_norm
        self.VALUE_COEF = value_coef
        self.BETA = beta
        self.GAMMA = gamma
        self.TMAX = tmax
        self.device = device

        self.vec_env = env
        self.network: A2C = A2C(n_act, network_size, shared_network)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)
        if environment_type == ENVS.pong:
            self.optimizer = optim.RMSprop(
                self.network.parameters(), lr=learning_rate, alpha=0.99, eps=1e-5
            )

        self.mean_comb_loss = []
        self.episode_steps = [0] * num_environments
        self.episode_rewards = [0.0] * num_environments
        self.final_lengths: list[list[int]] = [[] for _ in range(num_environments)]
        self.final_rewards = [[] for _ in range(num_environments)]
        self.comb_losses = [[] for _ in range(num_environments)]
        self.policy_losses = [[] for _ in range(num_environments)]
        self.value_losses = [[] for _ in range(num_environments)]
        self.entropy_losses = [[] for _ in range(num_environments)]
        self.env_type = environment_type
        self.preprocess_fn = preprocess_fn

    def preprocess_state(self, state) -> torch.Tensor:
        if self.env_type == ENVS.pong:
            frames: np.ndarray = (
                self.preprocess_fn(state) if self.preprocess_fn is not None else exit(1)
            )
            # Convert to float then normalize the uint8
            frames_t = torch.tensor(frames, dtype=torch.float, device=self.device)
            frames_t = frames_t / 255.0
            return frames_t

        return torch.tensor(state, device=self.device)

    def choose_actions(self, state: torch.Tensor):
        action_prob, values = self.network(state)
        distributions = Categorical(probs=action_prob)
        actions = distributions.sample()

        return (
            actions,
            distributions.log_prob(actions),
            distributions.entropy(),
            values,
        )

    def multiagent_loss(
        self,
        final_rewards: torch.Tensor,
        agent_experiences: list[list[Experience]],
        total_entropies: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        # Calculate losses for each agent
        for i in range(self.NENVS):
            real_rewards = []
            discounted_rewards = final_rewards[i]
            for experience in agent_experiences[i].__reversed__():
                if experience.done:
                    discounted_rewards = 0
                discounted_rewards = experience.reward + self.GAMMA * discounted_rewards
                real_rewards.insert(0, discounted_rewards)
            real_rewards_t = torch.tensor(real_rewards, device=self.device).unsqueeze(1)

            state_values = torch.stack([exp.value for exp in agent_experiences[i]]).to(
                device=self.device
            )
            advantages_t = real_rewards_t - state_values

            # Calculate Value loss
            value_loss = torch.pow(advantages_t, 2).mean() * self.VALUE_COEF

            log_probs = (
                torch.stack([exp.log_probability for exp in agent_experiences[i]])
                .to(device=self.device)
                .unsqueeze(1)
            )
            policy_loss = (-log_probs * advantages_t.detach()).mean()

            # Calculate Entropy loss
            # Mean entropy * beta
            entropy_loss = total_entropies[i] / len(agent_experiences[i]) * self.BETA

            # Calculate the combined loss
            combined_loss = policy_loss + value_loss - entropy_loss
            losses.append(combined_loss)

            # Add logging data
            self.comb_losses[i].append(combined_loss.item())
            self.value_losses[i].append(value_loss.item())
            self.policy_losses[i].append(policy_loss.item())
            self.entropy_losses[i].append(entropy_loss.item())

        # Combine the loss from each agent
        loss_t = torch.stack(losses).mean()
        # Add logging data
        self.mean_comb_loss.append(loss_t.item())
        return loss_t

    def parallel_rollout(self, initial_states: torch.Tensor):
        states = initial_states
        dones = np.array([False] * self.NENVS)

        experiences: list[list[Experience]] = [[] for _ in range(self.NENVS)]
        total_entropies: torch.Tensor = torch.tensor(
            [0] * self.NENVS, dtype=torch.float, device=self.device
        )

        # Run Rollout
        for t in range(self.TMAX):
            # print(states)
            actions, log_probs, entropies, values = self.choose_actions(states)
            total_entropies += entropies

            # Take actions
            observations, rewards, terminateds, truncateds, _ = self.vec_env.step(
                actions.numpy()
            )
            rewards = torch.tensor(rewards, device=self.device)
            # Calculate which
            dones = np.logical_or(terminateds, truncateds)

            for i in range(self.NENVS):
                # Add logging data
                self.episode_rewards[i] += rewards[i].item()
                self.episode_steps[i] += 0 if dones[i] else 1
                if terminateds[i] or truncateds[i]:
                    self.final_rewards[i].append(self.episode_rewards[i])
                    self.final_lengths[i].append(self.episode_steps[i])
                    self.episode_rewards[i] = 0
                    self.episode_steps[i] = 0
                # Add experience to replay memory
                experiences[i].append(
                    Experience(
                        states[i],
                        actions[i],
                        log_probs[i],
                        rewards[i],
                        values[i],
                        dones[i],
                    )
                )
            # Set next states
            states = self.preprocess_state(observations)

        # Mask final values with states which ended
        with torch.no_grad():
            _, final_values = self.network(states)
        final_values[dones] = 0

        # Back propogate loss
        losses = self.multiagent_loss(
            final_values.squeeze(1), experiences, total_entropies
        )
        self.optimizer.zero_grad()
        losses.backward()
        if self.env_type == ENVS.cartpole:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.CLIP_NORM)
        self.optimizer.step()

        return states

    def train(self, max_rollouts: int):
        obs, _ = self.vec_env.reset()
        states = self.preprocess_state(obs)

        for r in range(max_rollouts):
            # Visualize every 25 rollouts
            if r % 25 == 0:
                self.visualize(
                    "Mean Episode Length",
                    [sum(m) / len(m) for m in zip(*self.final_lengths)],
                    1,
                )
                self.multi_visualize("Combination Loss", self.comb_losses, 2)
                self.multi_visualize("Policy Loss", self.policy_losses, 3)
                self.multi_visualize("Value Loss", self.value_losses, 4)
                self.multi_visualize("Entropy Loss", self.entropy_losses, 5)
                self.visualize(
                    "Mean Cumulative Rewards",
                    [sum(m) / len(m) for m in zip(*self.final_rewards)],
                    6,
                )
                self.visualize("Mean Combination Loss", self.mean_comb_loss, 8)
            states = self.parallel_rollout(states)

    def multi_visualize(self, title: str, data: list, figure: int):
        plt.figure(figure)
        plt.clf()
        plt.title(title)

        plt.xlabel("Rollouts")
        plt.ylabel("Reward")
        for reward in data:
            reward = torch.tensor(reward, dtype=torch.float)
            plt.plot(reward.numpy())

            # Plot 100 episode averages
            if len(reward) > 100:
                means = reward.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())
                # if True:
                # plt.title(f"{title}, best 100 avg: {means.numpy().max()}")

        if plt.isinteractive():
            plt.pause(0.001)

    def visualize(self, title: str, data: list, figure: int):
        plt.figure(figure)
        rewards_t = torch.tensor(data, dtype=torch.float)
        plt.clf()
        plt.title(title)

        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.plot(rewards_t.numpy())

        # Plot 100 episode averages
        if len(rewards_t) > 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
            if True:
                plt.title(f"{title}, best 100 avg: {means.numpy().max()}")

        if plt.isinteractive():
            plt.pause(0.001)

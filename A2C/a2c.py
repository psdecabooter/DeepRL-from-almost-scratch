from collections import deque
from enum import Enum
import math
from queue import Queue
from typing import NamedTuple
import gymnasium as gym
from matplotlib import pyplot as plt
from numpy import dtype
from sympy import Transpose
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


class ENVS(Enum):
    cartpole = 0
    pong = 1


class A2C_algorithm:
    def __init__(
        self,
        shared_network: nn.Module,
        env: gym.Env,
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
        preprocess_fn = None
    ):
        # Hyperparameters
        self.CLIP_NORM = clip_norm
        self.VALUE_COEF = value_coef
        self.BETA = beta
        self.GAMMA = gamma
        self.TMAX = tmax
        self.device = device

        self.environment = env
        self.network: A2C = A2C(n_act, network_size, shared_network)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)

        self.episode_steps = 0
        self.episode_lengths: list[int] = []
        self.comb_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.env_type = environment_type
        self.preprocess_fn = preprocess_fn

    def preprocess_state(self, state) -> torch.Tensor:
        if self.env_type == ENVS.pong:
            frame = self.preprocess_fn(state) if self.preprocess_fn is not None else exit(1)
            return torch.tensor(frame, dtype=torch.uint8, device=self.device)

        return torch.tensor(state, device=self.device)

    def choose_action(self, state: torch.Tensor):
        action_probabilities, _ = self.network(state)
        # Choose an action from the distribution
        prob_distribution = Categorical(probs=action_probabilities)
        action = prob_distribution.sample()

        return (
            action.unsqueeze(0),
            prob_distribution.log_prob(action),
            prob_distribution.entropy().mean(),
        )

    def calculate_loss(
        self,
        final_reward: torch.Tensor,
        experiences: list[Experience],
        total_entropy: torch.Tensor,
    ) -> torch.Tensor:
        # Transpose the experiences
        # print(len(experiences))
        # Move backward through the experiences to calculate discounted rewards
        rewards = []
        discounted_rewards = final_reward
        for experience in experiences.__reversed__():
            # Accumulate rewards
            discounted_rewards = experience.reward + self.GAMMA * discounted_rewards
            rewards.insert(0, discounted_rewards)
        rewards_t = torch.stack(rewards).to(self.device)
        # print(rewards_t)

        states_t = torch.stack([exp.state for exp in experiences]).to(self.device)
        # print("states: ", states_t.shape)
        _, expected_values = self.network(states_t)
        # print("expctV: ", expected_values)

        # Advantage is the difference between the expected and real reward
        advantage: torch.Tensor = rewards_t - expected_values
        # print("adv: ", advantage)

        # Convert log probabilities to tensor
        log_probabilities = (
            torch.stack([exp.log_probability for exp in experiences])
            .to(self.device)
            .unsqueeze(-1)
        )
        # print("lp: ", log_probabilities.shape)
        # print(log_probabilities)

        # Calculate Value Loss using mean squared error of the advantage
        # want the gradient wrt value parameters
        criterion = nn.MSELoss()
        value_loss: torch.Tensor = criterion(expected_values, rewards_t)

        # Calculate policy loss using the expected value of the product of the log probabilities
        # of the actions taken and the advantage
        # Only want the gradient wrt policy parameters
        policy_loss = (-log_probabilities * advantage.detach()).mean()
        # print("ploss: ", policy_loss)
        # print(policy_loss)

        # Combined loss plus entropy
        mean_entropy = total_entropy / len(experiences)
        # print("ment: ", mean_entropy)
        comb_loss = (
            policy_loss + self.VALUE_COEF * value_loss - self.BETA * mean_entropy
        )
        self.comb_losses.append(comb_loss.item())
        self.value_losses.append(self.VALUE_COEF * value_loss.item())
        self.policy_losses.append(policy_loss.item())
        self.entropy_losses.append(-self.BETA * mean_entropy.item())
        # print("vloss: ", value_loss)
        # print("closs: ", comb_loss)
        # exit()
        return comb_loss

    def train_nstep_rollout(self, initial_state: torch.Tensor) -> torch.Tensor | None:
        state: torch.Tensor = initial_state

        early_terminate: bool = False

        experiences: list[Experience] = []
        total_entropy: torch.Tensor = torch.tensor(
            0, dtype=torch.float, device=self.device
        )
        for t in range(self.TMAX):
            self.episode_steps += 1
            action, log_prob, entropy = self.choose_action(state)
            # print("Act: ", action.shape)
            # print(action)
            # print("e: ", entropy)
            total_entropy += entropy

            observation, reward, terminated, truncated, _ = self.environment.step(
                action.item()
            )

            reward = torch.tensor(reward, device=self.device)

            if terminated or truncated:
                early_terminate = True
                experiences.append(Experience(state, action, log_prob, reward))
                break
            else:
                next_state = self.preprocess_state(observation)
                experiences.append(Experience(state, action, log_prob, reward))
                state = next_state

        # Initialize total rewards to the value of the current state
        with torch.no_grad():
            _, final_value = self.network(state)
        # If it early terminated, there are no future rewards
        if early_terminate:
            final_value = torch.zeros_like(final_value, device=self.device)
        else:
            final_value.detach()

        # Calculate n-step loss
        loss = self.calculate_loss(final_value, experiences, total_entropy)
        # print("loss: ", loss.shape)
        # print(loss)

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.CLIP_NORM)
        self.optimizer.step()
        return None if early_terminate else state

    def train(self, max_rollouts: int):
        obs, _ = self.environment.reset()
        state = self.preprocess_state(obs)
        episodes = 0

        for _ in range(max_rollouts):
            next_state = self.train_nstep_rollout(state)
            if next_state is None:
                episodes += 1
                # Find new state
                obs, _ = self.environment.reset()
                state = self.preprocess_state(obs)
                # print(self.episode_steps)
                self.episode_lengths.append(self.episode_steps)
                self.episode_steps = 0
                print(episodes)
            else:
                state = next_state

    def visualize(self, title: str, data: list, figure: int):
        plt.figure(figure)
        rewards_t = torch.tensor(data, dtype=torch.float)
        plt.clf()
        if True:
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
            if True:
                plt.title(f"{title}, best 100 avg: {means.numpy().max()}")

        if plt.isinteractive():
            plt.pause(0.001)

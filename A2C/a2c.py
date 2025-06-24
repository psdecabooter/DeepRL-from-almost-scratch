import math
from typing import NamedTuple
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class A2C(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super(A2C, self).__init__()

        SIZE: int = 128

        # Shared Parameters
        self.main = nn.Sequential(
            nn.Linear(observation_size, SIZE),
            nn.ReLU(),
            nn.Linear(SIZE, SIZE),
            nn.ReLU(),
        )

        self.policy = nn.Linear(SIZE, action_size)
        self.value = nn.Linear(SIZE, 1)

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
    log_probability: float
    reward: torch.Tensor
    next_state: torch.Tensor


class A2C_algorithm:
    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        n_obsv: int,
        n_act: int,
        beta: float,
        gamma: float,
        learning_rate: float,
        value_coef: float,
        tmax: int,
        clip_norm: float,
    ):
        # Hyperparameters
        self.CLIP_NORM = clip_norm
        self.VALUE_COEF = value_coef
        self.BETA = beta
        self.GAMMA = gamma
        self.TMAX = tmax
        self.device = device

        self.environment = env
        self.network: A2C = A2C(n_obsv, n_act)

        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)

    def preprocess_state(self, state) -> torch.Tensor:
        return torch.tensor(state, device=self.device)

    def choose_action(self, action_probabilities: torch.Tensor):
        # Choose an action from the distribution
        prob_distribution = Categorical(probs=action_probabilities)
        action = prob_distribution.sample()

        return (
            action,
            prob_distribution.log_prob(action),
            prob_distribution.entropy().mean(),
        )

    def calculate_loss(
        self,
        initial_state: torch.Tensor,
        final_reward: torch.Tensor,
        experiences: list[Experience],
        total_entropy: torch.Tensor,
    ) -> torch.Tensor:
        # Move backward through the experiences to calculate discounted rewards
        discounted_rewards = final_reward
        for experience in experiences.__reversed__():
            # Accumulate rewards
            discounted_rewards = experience.reward + self.GAMMA * discounted_rewards

        # Advantage is the difference between the expected and real reward
        _, expected_reward = self.network(initial_state)
        advantage: torch.Tensor = discounted_rewards - expected_reward

        # Convert log probabilities to tensor
        log_probabilities = torch.tensor(
            [exp.log_probability for exp in experiences], device=self.device
        )

        # Calculate Value Loss using mean squared error of the advantage
        # want the gradient wrt value parameters
        value_loss = advantage.pow(2).mean()

        # Calculate policy loss using the expected value of the product of the log probabilities
        # of the actions taken and the advantage
        # Only want the gradient wrt policy parameters
        policy_loss = (-log_probabilities * advantage.detach()).mean()

        # Combined loss plus entropy
        comb_loss = (
            policy_loss + self.VALUE_COEF * value_loss + self.BETA * total_entropy
        )
        return comb_loss

    def train_nstep_rollout(self, initial_state: torch.Tensor) -> torch.Tensor | None:
        state: torch.Tensor = initial_state

        early_terminate: bool = False

        experiences: list[Experience] = []
        total_entropy: torch.Tensor = torch.zeros(device=self.device)
        for t in range(self.TMAX):
            with torch.no_grad():
                action_probabilities, _ = self.network(state)
                action, log_prob, entropy = self.choose_action(action_probabilities)
            total_entropy += entropy

            observation, reward, terminated, truncated, _ = self.environment.step(
                action
            )

            reward = torch.tensor(reward, device=self.device)

            if terminated or truncated:
                early_terminate = True
                break
            else:
                next_state = self.preprocess_state(observation)
                experiences.append(
                    Experience(state, action, log_prob, reward, next_state)
                )
                state = next_state

        # Initialize total rewards to the value of the current state
        with torch.no_grad():
            _, final_value = self.network(state)
        # If it early terminated, there are no future rewards
        if early_terminate:
            final_value = torch.zeros_like(final_value, device=self.device)

        # Calculate n-step loss
        loss = self.calculate_loss(
            initial_state, final_value, experiences, total_entropy
        )

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.CLIP_NORM)
        self.optimizer.step()
        return None if early_terminate else state

    def train(self, rollouts: int):
        obs, _ = self.environment.reset()
        state = self.preprocess_state(obs)

        for u in range(rollouts):
            next_state = self.train_nstep_rollout(state)
            if next_state is None:
                obs, _ = self.environment.reset()
                state = self.preprocess_state(obs)
            else:
                state = next_state

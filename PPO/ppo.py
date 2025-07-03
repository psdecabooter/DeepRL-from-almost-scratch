from importlib.metadata import distribution
from typing import NamedTuple
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym


class PPONetwork(nn.Module):
    def __init__(
        self, network_size: int, action_space: int, shared_parameters: nn.Module
    ):
        super(PPONetwork, self).__init__()

        self.shared_parameters = shared_parameters

        self.policy_funciton = nn.Linear(network_size, action_space)
        self.value_function = nn.Linear(network_size, 1)

    def forward(self, state: torch.Tensor):
        shared = self.shared_parameters(state)
        policy = self.policy_funciton(shared)
        value = self.value_function(shared)
        return F.softmax(policy, dim=-1), value


class Experience(NamedTuple):
    """
    Holds Replay experience from rollouts
    Tensors are batched by the number of actors
    """

    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    state_values: torch.Tensor
    rewards: torch.Tensor
    entropies: torch.Tensor
    dones: list[bool]


class PPOClient(object):
    def __init__(
        self,
        actor_num: int,
        horizon: int,
        minibatch_size: int,
        action_space: int,
        network_size: int,
        epochs: int,
        gamma: float,
        epsilon: float,
        value_coefficient: float,
        entropy_coefficient: float,
        clip_norm: float,
        learning_rate: float,
        network: nn.Module,
        environments: gym.vector.VectorEnv,
        device: torch.device,
    ):
        # Hyper parameters
        self.HORIZON = horizon
        self.ACTORS = actor_num
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPOCHS = epochs
        self.VCOEF = value_coefficient
        self.ECOEF = entropy_coefficient
        self.CLIP_NORM = clip_norm
        self.MINIBATCH_SIZE = minibatch_size

        # Other
        self.device = device
        self.network = PPONetwork(network_size, action_space, network)
        self.env = environments
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)

        # Logging information
        self.policy_losses = []
        self.value_losses = []
        self.combined_losses = []
        self.cum_rewards = []

        # Ensure valid hyperparameters
        assert self.MINIBATCH_SIZE <= self.HORIZON * self.ACTORS
        assert self.HORIZON % self.MINIBATCH_SIZE == 0

    def choose_actions(
        self, action_distributions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        action_probs has shape (N, A)
        Where N is the number of actors and A is the action space of the environment

        Returns:
            Tensor: actions sampled from action probs
            Tensor: the log probabilities of the actions sampled
        """
        distributions = Categorical(probs=action_distributions)
        actions = distributions.sample()
        return (
            actions.unsqueeze(1),
            distributions.log_prob(actions).unsqueeze(1),
            distributions.entropy().unsqueeze(1),
        )

    def rollout(self, starting_states: torch.Tensor):
        states = starting_states
        dones = np.array([False] * self.ACTORS)
        cumulative_rewards = torch.zeros(self.ACTORS)

        # Hold Replay Memory
        # T x N x ...
        replay_experiences = []
        for t in range(self.HORIZON):
            with torch.no_grad():
                action_distributions, state_values = self.network(states)
            actions, log_probs, entropies = self.choose_actions(action_distributions)

            next_states, rewards, terminateds, truncateds, _ = self.env.step(
                actions.squeeze(-1).numpy()
            )
            rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
            cumulative_rewards += rewards.squeeze(1)
            dones = np.logical_or(terminateds, truncateds)

            # Record experiences
            replay_experiences.append(
                Experience(
                    states,
                    actions,
                    log_probs,
                    state_values,
                    rewards,
                    entropies,
                    dones,
                )
            )

            states = next_states

            # Record logging data
            self.cum_rewards.extend(cumulative_rewards[dones])
            cumulative_rewards[dones] = 0

        # Calculate final state-values
        with torch.no_grad():
            _, final_state_values = self.network(states)
        # Zero out terminated states
        final_state_values[dones] = 0

        # Return final states, final state values, and the replay experience
        return states, final_state_values, replay_experiences

    def pp_optimize(self, experiences: list[Experience], final_values: torch.Tensor):
        """
        Calculates PPO Loss and trains it for K epochs
        Batches are based on number of actors
        The expectation indicates the empirical average over a finite batch of samples
        """

        # calculate advantages
        advantages = []
        target_values = []
        discounted_rewards = final_values
        for experience in experiences.__reversed__():
            discounted_rewards[experience.dones] = 0
            discounted_rewards = experience.rewards + self.GAMMA * discounted_rewards
            advantages.insert(0, discounted_rewards - experience.state_values)
            target_values.insert(0, discounted_rewards)

        # Create batches
        advantages_t = torch.cat(advantages).to(device=self.device)
        states_t = torch.cat([exp.states for exp in experiences]).to(device=self.device)
        actions_t = torch.cat([exp.actions for exp in experiences]).to(
            device=self.device
        )
        log_probs_t = torch.cat([exp.log_probs for exp in experiences]).to(
            device=self.device
        )
        target_values_t = torch.cat(target_values).to(device=self.device)

        # Ensure that they are all the same size
        assert (
            len(advantages_t)
            == len(states_t)
            == len(actions_t)
            == len(log_probs_t)
            == len(target_values_t)
        )

        # Go through batches of experiences in a random order
        order = np.arange(self.HORIZON * self.ACTORS)
        np.random.shuffle(order)
        for start in range(0, len(order), self.MINIBATCH_SIZE):
            end = min(len(order), start + self.MINIBATCH_SIZE)
            batch_indicies = order[start:end]
            # Perform learning on minibatch
            action_probs, new_state_values = self.network(states_t[batch_indicies])
            action_dist = Categorical(probs=action_probs)
            entropies = action_dist.entropy()
            new_log_probs = action_dist.log_prob(
                actions_t[batch_indicies].squeeze(1)
            ).unsqueeze(1)

            # Calculate Probability Ratio
            p_ratio = torch.exp(new_log_probs - log_probs_t[batch_indicies])
            clipped_ratio = torch.clip(p_ratio, 1 - self.EPSILON, 1 + self.EPSILON)

            # Calculate the value loss
            # MSE of new state values - target values
            # Take the mean since the expectation is the empirical average over
            #   a finite batch of samples
            value_loss = (
                0.5
                * torch.pow(
                    new_state_values - target_values_t[batch_indicies], 2
                ).mean()
            )

            # Calculate Policy loss
            # Negative since im doing gradient descent not ascent
            # Take the mean since the expectation is the empirical average over
            #   a finite batch of samples
            policy_loss = (
                -1
                * torch.min(
                    p_ratio * advantages_t[batch_indicies],
                    clipped_ratio * advantages_t[batch_indicies],
                ).mean()
            )

            # Calculate Entropy loss
            entropy_loss = entropies.mean()

            # Combine losses
            # Take the mean since the expectation is the empirical average over
            #   a finite batch of samples
            combined_loss = (
                policy_loss + self.VCOEF * value_loss - self.ECOEF * entropy_loss
            )

            self.optimizer.zero_grad()
            combined_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.CLIP_NORM)
            self.optimizer.step()

            # Add logging information
            self.policy_losses.append(policy_loss)
            self.value_losses.append(self.VCOEF * value_loss)
            self.combined_losses.append(combined_loss)

    def train(self, iterations: int):
        states, _ = self.env.reset()
        for i in range(iterations):
            states, final_state_values, experiences = self.rollout(states)
            self.pp_optimize(experiences, final_state_values)

            # Update graphs
            if i % 1 == 0:
                self.visualize("Cumulative Rewards", self.cum_rewards, 1)
                self.visualize("Policy Losses", self.policy_losses, 2)
                self.visualize("Value Losses", self.value_losses, 3)
                self.visualize("Combined Losses", self.combined_losses, 4)

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

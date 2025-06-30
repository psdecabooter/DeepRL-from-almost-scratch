import datetime
from itertools import count
import os
import random
import time
import gymnasium as gym
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from a2c import A2C_algorithm


def main():
    NUM_ENVS = 12
    env = gym.make_vec("CartPole-v1", NUM_ENVS, vectorization_mode="sync")
    # env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    observation_size = env.single_observation_space.shape[0]  # type: ignore
    action_size = env.single_action_space.n  # type: ignore
    print(action_size)
    print(observation_size)

    # if GPU is to be used
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Hyper Parameters
    GAMMA = 0.99  # Controls reward decay
    LEARNING_RATE = 0.0004  # Learning rate of AdamW Optimizer
    VALUE_COEFFICIENT = 0.5  # policy_loss + VAL_COEF * value_loss
    CLIP_NORM = 0.1  # Parameter clipping
    BETA = 0.05  # Controls how important entropy is
    ROLLOUT_LENGTH = 20  # Length of rollouts
    ROLLOUTS = 5000  # Number of episodes
    NETSIZE = 64

    # Shared network which the a2c will use
    shared_network = nn.Sequential(
        nn.Linear(observation_size, NETSIZE),
        nn.ReLU(),
        nn.Linear(NETSIZE, NETSIZE),
        nn.ReLU(),
    )
    # A2C client
    a2c = A2C_algorithm(
        num_environments=NUM_ENVS,
        shared_network=shared_network,
        env=env,
        device=device,
        n_act=action_size,
        beta=BETA,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        value_coef=VALUE_COEFFICIENT,
        tmax=ROLLOUT_LENGTH,
        clip_norm=CLIP_NORM,
        network_size=NETSIZE,
    )

    # seed = 42
    # random.seed(seed)
    # torch.manual_seed(seed)
    # env.reset(seed=seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

    plt.ion()
    a2c.train(ROLLOUTS)
    plt.ioff()
    # a2c.multi_visualize("Episode Length", a2c.episode_lengths, 1)
    # a2c.multi_visualize("Combination Loss", a2c.comb_losses, 2)
    # a2c.multi_visualize("Policy Loss", a2c.policy_losses, 3)
    # a2c.multi_visualize("Value Loss", a2c.value_losses, 4)
    # a2c.multi_visualize("Entropy Loss", a2c.entropy_losses, 5)

    env = gym.make("CartPole-v1")
    rewards = []
    for i in range(500):
        terminated = False
        state, _ = env.reset()
        episode_reward = 0

        for i in count():
            state_t = a2c.preprocess_state(state)
            # print("s: ", state_t.shape)

            with torch.no_grad():
                action_probs, expected_value = a2c.network(state_t)
                action = torch.argmax(action_probs).item()
            # print(action_probs.shape)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state

            # Render and add delay for visualization
            # env.render()
            # time.sleep(0.02)

            if terminated or truncated:
                episode_reward = i
                break

        rewards.append(episode_reward)
    a2c.visualize("Trained Policy 1k Test", rewards, 6)
    plt.show()


if __name__ == "__main__":
    main()

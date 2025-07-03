from itertools import count
import gymnasium as gym
from gymnasium.vector import VectorObservationWrapper
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from ppo import PPOClient


class FLPreprocessObs(VectorObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observations(self, observations):
        obs_size = self.single_observation_space.n  # type: ignore
        state = torch.zeros(
            (self.num_envs, obs_size),
            dtype=torch.float,
        )
        state[np.arange(self.num_envs), observations] = 1
        return state


class FrozenNetwork(nn.Module):
    def __init__(self, observation_size: int, network_size: int) -> None:
        super(FrozenNetwork, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(observation_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.main(state)


def main():
    # HYPER PARAMETERS
    LEARNING_RATE = 2e-4
    ITERATIONS = 200
    HORIZON = 2048
    BATCH_SIZE = 64
    EPOCHS = 10
    GAMMA = 0.99
    CLIP_RANGE = 0.2
    CLIP_NORM = 0.5
    VALUE_COEFFICIENT = 0.5
    ENTROPY_COEFFICIENT = 0.01
    NUM_ENVS = 6
    NETSIZE = 128

    # Environment
    envs = gym.make_vec("FrozenLake-v1", num_envs=NUM_ENVS)
    envs = FLPreprocessObs(envs)
    observation_size = envs.single_observation_space.n  # type: ignore
    action_size = envs.single_action_space.n  # type: ignore

    # Network
    network = FrozenNetwork(observation_size, NETSIZE)

    # if GPU is to be used
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # PPO Client
    ppo = PPOClient(
        actor_num=NUM_ENVS,
        horizon=HORIZON,
        minibatch_size=BATCH_SIZE,
        action_space=action_size,
        network_size=NETSIZE,
        epochs=EPOCHS,
        gamma=GAMMA,
        epsilon=CLIP_RANGE,
        value_coefficient=VALUE_COEFFICIENT,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        clip_norm=CLIP_NORM,
        learning_rate=LEARNING_RATE,
        network=network,
        environments=envs,
        device=device,
    )

    plt.ion()
    ppo.train(ITERATIONS)
    plt.ioff()

    # Test trained model
    env = gym.make("FrozenLake-v1")
    final_rewards = []
    for e in range(500):
        start_state, _ = env.reset()
        state = torch.zeros(observation_size)
        state[start_state] = 1
        for i in count():
            with torch.no_grad():
                action_probs, state_values = ppo.network(state)
                action = torch.argmax(action_probs).item()
            next_state, reward, term, trun, _ = env.step(action)
            state = torch.zeros(observation_size)
            state[next_state] = 1
            if term or trun:
                final_rewards.append(reward)
                break

    ppo.visualize("Trained Final Rewards", final_rewards, 0)
    plt.show()

    # Visualize model on loop
    env = gym.make("FrozenLake-v1", render_mode="human")
    for _ in count():
        start_state, _ = env.reset()
        state = torch.zeros(observation_size)
        state[start_state] = 1
        for i in count():
            with torch.no_grad():
                action_probs, state_values = ppo.network(state)
                action = torch.argmax(action_probs).item()
            next_state, reward, term, trun, _ = env.step(action)
            state = torch.zeros(observation_size)
            state[next_state] = 1
            if term or trun:
                break

    env.close()
    envs.close()


if __name__ == "__main__":
    main()

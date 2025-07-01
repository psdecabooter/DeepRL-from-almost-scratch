import datetime
from itertools import count
import os
import random
import time
import ale_py.vector_env
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import ale_py
from a2c import ENVS, A2C_algorithm
import numpy as np
import cv2


class PongArchitecture(nn.Module):
    def __init__(self, obs_shape, action_size: int, net_size: int):
        super(PongArchitecture, self).__init__()
        # The input is a 84x84x4 image
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_shape[0],
                out_channels=32,
                stride=4,
                kernel_size=8,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                stride=1,
                kernel_size=3,
            ),
            nn.ReLU(),
        )

        conv_out_size = np.prod(self.conv(torch.rand(1, *obs_shape)).data.shape)

        self.final = nn.Sequential(
            nn.Linear(conv_out_size, net_size),
            nn.ReLU(),
        )

    def forward(self, state):
        state = self.conv(state)
        # print("fhd", state.shape)
        state = state.view(-1, 7 * 7 * 64)
        return self.final(state)


def preprocess_state(frame: np.ndarray):
    # Take in a 210 x 160 pixel grayscale image
    state = np.ndarray((4, 4, 84, 84))
    for g in range(len(frame)):
        for i in range(len(frame[g])):
            crop = frame[g][i][13:-6, 3:-2]
            # print(type(frame))
            state[g][i] = np.array(
                cv2.resize(crop, (84, 84), interpolation=cv2.INTER_NEAREST),
                dtype=np.uint8,
            )
    return state


def main():
    NENVS = 4
    print("back: ", plt.get_backend())
    gym.register_envs(ale_py)
    # env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    # env = FrameStackObservation(env, stack_size=4)
    env = ale_py.vector_env.AtariVectorEnv(game="pong", num_envs=NENVS)
    obs, _ = env.reset()
    action_size = env.single_action_space.n  # type: ignore
    # action_size = env.action_space.n  # type: ignore
    # print(action_size)

    # print(obs)
    # print(obs.shape)
    # obs = preprocess_state(obs)
    # print(obs)
    # print(obs.shape)
    # exit()
    # For visualizing
    # plt.ion()
    # plt.figure(figsize=(10, 5))
    # plt.gcf().set_facecolor("lightgreen")
    # for i in range(40):
    #     obs, _, _, _, _ = env.step(env.action_space.sample())  # type: ignore
    #     plt.subplot(1, 2, 1)
    #     plt.title("Raw Frame (RGB)")
    #     plt.imshow(obs[0][1])
    #     plt.axis("off")
    #     plt.subplot(1, 2, 2)
    #     plt.title("Processed Frame")
    #     plt.imshow(preprocess_state(obs)[0][1])
    #     plt.axis("off")
    #     if plt.isinteractive():
    #         plt.pause(0.001)
    # plt.ioff()
    # exit()

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
    LEARNING_RATE = 7e-4  # Learning rate of AdamW Optimizer
    VALUE_COEFFICIENT = 0.5  # policy_loss + VAL_COEF * value_loss
    CLIP_NORM = 0.1  # Parameter clipping
    BETA = 0.01  # Controls how important entropy is
    ROLLOUT_LENGTH = 5  # Length of rollouts
    ROLLOUTS = 250000  # Number of episodes
    NETSIZE = 256
    SHARED_NETWORK = PongArchitecture(obs[0].shape, action_size, NETSIZE)
    # A2C client
    a2c = A2C_algorithm(
        num_environments=4,
        shared_network=SHARED_NETWORK,
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
        environment_type=ENVS.pong,
        preprocess_fn=preprocess_state,
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
    a2c.multi_visualize("Episode Length", a2c.final_lengths, 1)
    a2c.multi_visualize("Combination Loss", a2c.comb_losses, 2)
    a2c.multi_visualize("Policy Loss", a2c.policy_losses, 3)
    a2c.multi_visualize("Value Loss", a2c.value_losses, 4)
    a2c.multi_visualize("Entropy Loss", a2c.entropy_losses, 5)

    # Save policy
    save_dict = {"model_state_dict": a2c.network.state_dict()}
    torch.save(save_dict, "pong_policy.pth")

    rewards = []
    for i in range(500):
        terminated = False
        state, _ = env.reset()
        episode_reward = 0

        for i in count():
            state_t = a2c.preprocess_state(state)

            with torch.no_grad():
                action_probs, expected_value = a2c.network(state_t)
                action = torch.argmax(action_probs).item()

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)  # type: ignore
            state = next_state

            # Render and add delay for visualization
            # env.render()
            # time.sleep(0.02)

            if terminated or truncated:
                episode_reward = i
                break

        rewards.append(episode_reward)
    a2c.multi_visualize("Trained Policy 1k Test", rewards, 100)
    plt.show()

    env.close()


if __name__ == "__main__":
    main()

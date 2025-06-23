import datetime
from itertools import count
import os
import time
import gymnasium as gym
from matplotlib import pyplot as plt
import torch
import torch.optim as optim

from dqn_scratch import (
    DQN,
    Envs,
    EpsilonGreedyPolicy,
    ReplayBuffer,
    perform_DQN_episodes,
    preprocess_state,
    visualize_episodes,
)


def main():
    # Set up plot
    fig, axes = plt.subplots()

    # env = gym.make("FrozenLake-v1", render_mode="human")
    env = gym.make("CartPole-v1")

    # if GPU is to be used
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # HYPER PARAMETERS
    ENVIRONMNET = Envs.CARTPOLE
    PLOT = True
    LABEL = str(int(time.time() * 10))
    BATCH_SIZE = 128  # the number of transitions sampled from the replay buffer
    GAMMA = 0.99  # the discount factor
    EPS_START = 0.9  # the starting epsilon greedy parameter
    EPS_END = 0.01  # the ending epsilon greedy parameter
    EPS_DECAY = 0.9995  # controls the rate of exponential decay of epsilon, higher means slower decay
    TAU = 0.005  # the update rate of the target network
    LR = 3e-4  # the learning rate of the AdamW optimizer

    # If the label directory doesn't exist and plt is active, make it
    if PLOT and not os.path.exists(LABEL):
        os.makedirs(LABEL)

    n_actions = env.action_space.n  # type: ignore
    # get the number of state observations
    state, _ = env.reset()
    n_observations = len(state)  # type: ignore

    egreedy = EpsilonGreedyPolicy(
        device=device, lower=EPS_END, upper=EPS_START, decay_rate=EPS_DECAY
    )
    policy_net = DQN(n_actions, n_observations).to(device)
    target_net = DQN(n_actions, n_observations).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(10000)

    num_episodes: int = 600
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 10000

    plt.ion()
    perform_DQN_episodes(
        state_size=n_observations,
        episodes=num_episodes,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        env=env,
        target=target_net,
        policy=policy_net,
        egreedy=egreedy,
        replay_buffer=memory,
        optimizer=optimizer,
        device=device,
        plot=PLOT,
        dtype=torch.float,
        label=LABEL,
        environment=ENVIRONMNET,
    )
    plt.ioff()
    env.close()

    print("Complete")

    # Save policy
    save_dict = {"model_state_dict": policy_net.state_dict()}
    torch.save(save_dict, "policy.pth")

    if not PLOT:
        return
    # env = gym.make("FrozenLake-v1", render_mode="human")
    # if plot, run 100 runs with the policy
    rewards = []
    for i in range(1000):
        terminated = False
        state, _ = env.reset()
        episode_reward = 0

        for i in count():
            state_t = preprocess_state(
                state, n_observations, device, torch.float, environment=ENVIRONMNET
            )

            with torch.no_grad():
                action_probs = policy_net(state_t)
                action = torch.argmax(action_probs).item()

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

    print(rewards)
    print(sum(rewards) / len(rewards))
    visualize_episodes(
        os.path.join(LABEL, "target_rewards.png"), rewards, torch.float, fig_num=3
    )

    env = gym.make("CartPole-v1", render_mode="human")
    while True:
        terminated = False
        state, _ = env.reset()

        while True:
            state_t = preprocess_state(
                state, n_observations, device, torch.float, environment=ENVIRONMNET
            )

            with torch.no_grad():
                action_probs = policy_net(state_t)
                action = torch.argmax(action_probs).item()

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state

            # Render and add delay for visualization
            # env.render()
            # time.sleep(0.02)

            if terminated or truncated:
                break


if __name__ == "__main__":
    main()

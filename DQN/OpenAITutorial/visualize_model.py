import gymnasium as gym
import torch
import time
from policy_utils import load_policy, DQN

env = gym.make("CartPole-v1", render_mode="human")

policy, info = load_policy("dqn_policy.pth", DQN, 4, 2)
if policy is None:
    exit()

while True:
    state, _ = env.reset()
    episode_reward = 0
    terminated = False

    episode = 0
    while not terminated:
        episode += 1
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        
        while not terminated:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy(state_tensor)
                action = torch.argmax(action_probs).item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward # type: ignore
            state = next_state
            
            # Render and add delay for visualization
            env.render()
            time.sleep(0.02)
            
            if terminated or truncated:
                print(f"Episode {episode + 1}: Reward = {episode_reward}")
                break
    
    env.close()
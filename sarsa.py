import gymnasium as gym
import torch 

env = gym.make('CartPole-v1')

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

print(f"\nNumber of actions: {env.action_space.n}")
print(f"State dimensions: {env.observation_space.shape}")
print(f"State bounds: {env.observation_space.low} to {env.observation_space.high}")

# training loop 
for episode in range(5):
    obs, info = env.reset()
    rewards = []
    done = False

    while not done:
        action = env.action_space.sample() #Random POLicy
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    print(f"all rewards: {rewards}")

env.close()
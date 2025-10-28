import gymnasium as gym
import torch 
import numpy as np
from tilecoding import TileCoding


class SarsaOne:
    def __init__(self,env:gym.envs, Tiles:TileCoding, gamma:float, epsilon:float,alpha:float, maxIter:int)->None:
        self.env = env
        self.tiles = Tiles
        self.g = gamma
        self.e = epsilon
        self.a = alpha
        self.maxIter = maxIter
        self.numActions = self.env.action_space.n

        self.weights = np.zeros(self.tiles.N)
    
    def get_q_value(self,state_features:np.ndarray, action):
        q = 0.0
        for feature_idx in state_features:
            q += self.weights[feature_idx]
        return q 

    def select_action(self, state_features:np.ndarray):
        if np.random.random() < self.e:
            return self.env.action_space.sample()
        qs = [self.get_q_value(state_features, a) for a in range(self.numActions)]
        return np.argmax(qs)
    
    def update(self, state_features, action, reward, next_state_features, next_action):
        qC = self.get_q_value(state_features, action)
        qN = self.get_q_value(next_state_features, next_action)\
        
        td = reward + self.g * qN - qC
        for feature_idx in state_features:
            self.weights[feature_idx] += self.a * td

    def decayEpsilon(self, episode, total_episodes):
        self.e = max(0.01, 1.0 * (1 - episode / total_episodes))

#driver code

env = gym.make("CartPole-v1")
tileCoding = TileCoding(10,4,np.array([-2.4,-1.0,-0.2,-1.0]), np.array([2.4,1.0,0.2,1.0]),1024)

agent = SarsaOne(env, tileCoding,0.99,0.2,0.1,1000)
episodeRewards = []
for episode in range(1000):
    obs, info = env.reset()
    stateFeatures = tileCoding.tileIndices(obs)
    action = agent.select_action(stateFeatures)

    episodeReward = 0
    done = False

    while not done:
        obs, reward, terminated, truncated, _ = env.step(action)
        nextStateFeatures = tileCoding.tileIndices(obs)
        nextAction = agent.select_action(nextStateFeatures)

        agent.update(stateFeatures, action, reward, nextStateFeatures, nextAction)

        episodeReward += reward
        stateFeatures = nextStateFeatures
        action = nextAction
        done = terminated or truncated
    episodeRewards.append(episodeReward)
    agent.decayEpsilon(episode, 1000)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episodeRewards[-10:])
        print(f"Episode {episode + 1}: Avg Reward (last 10) = {avg_reward:.2f}, ")
        print(f"Epsilon = {agent.e:.3f}")

env.close()
print("Training Complete!")

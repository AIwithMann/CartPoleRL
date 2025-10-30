import numpy as np 
import gymnasium as gym
import math as m
from tilecoding import TileCoding
class SarsaN:
    def __init__(self, env:gym.envs, tc:TileCoding, gamma:float, epsilon:float, alpha:float, maxIter:int, n:int):
        self.env = env
        self.tileCoding = tc
        self.g = gamma
        self.e = epsilon
        self.a = alpha
        self.maxIter = maxIter
        self.nSteps = n

        self.numActions = self.env.action_space.n
        self.weights = np.zeros(self.tileCoding.N * self.numActions)

    def Q(self,stateFeatures:list, action:int)->float:
        offset = m.floor((action * self.tileCoding.N)/self.numActions)
        q = 0.0
        for featureIdx in stateFeatures:
            q += self.weights[offset + featureIdx]
        return q

    def selectAction(self, stateFeatures:list):
        if np.random.random() < self.e:
            return self.env.action_space.sample()
        qs = [self.Q(stateFeatures,a) for a in range(self.numActions)]
        return np.argmax(qs)
    
    def update(self, stateFeatures:list, action:int, G:float, tauStateFeatures:list,tauAction:int):
        qC = self.Q(stateFeatures,action)
        qT = self.Q(tauStateFeatures,tauAction)
        td = G + self.g ** self.nSteps * qT 
        td = np.clip(td,-10,10)

        offset = m.floor((action * self.tileCoding.N)/self.numActions)
        for featureIdx in stateFeatures:
            self.weights[offset + featureIdx] += self.a * td 
    
    def decayEpsilon(self, episode, totalEpisodes):
        self.e = max(0.01, (1 - episode / totalEpisodes))
    
def getFinalAverage(gamma, alpha, epsilon, N):
    env = gym.make("CartPole-v1")
    print("done")
    low = np.array([-4.8, -5.0, -0.418,-5.0])
    high = np.array([4.8, 5.0, 0.418, 5.0])
    tc = TileCoding(10,4,low,high,8192)

    agent = SarsaN(env, tc, gamma,epsilon, alpha, 1000, N) 
    episodeRewards = []
    for episode in range(1000):
        states = []
        actions = []
        rewards = [None]
        
        obs, _ = env.reset()
        stateFeatures = tc.tileIndices(obs)
        a = agent.selectAction(stateFeatures)
        
        states.append(stateFeatures)
        actions.append(a)

        done = False 
        
        t = 0
        T = float('inf')
        while True:
            if t < T:
                obs, reward, terminated, truncated, _ = env.step(actions[-1])
                obs = tc.tileIndices(obs)
                states.append(obs)
                rewards.append(reward)

                if terminated or truncated:
                    T = t+1
                else:
                    actions.append(agent.selectAction(obs))
            
            tau = t - agent.nSteps + 1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+agent.nSteps, T+1)):
                    G += agent.g ** (i-tau-1) * rewards[i]
                
                if tau + agent.nSteps < T:
                    sTauN = states[tau+agent.nSteps]
                    aTauN = actions[tau+agent.nSteps]
                    G += (agent.g ** agent.nSteps) * agent.Q(sTauN, aTauN)
                sTau = states[tau]
                aTau = actions[tau]
                
                offset = m.floor((aTau * agent.tileCoding.N)/agent.numActions)
                for featureIdx in sTau:
                    agent.weights[offset + featureIdx] += agent.a * (G - agent.Q(sTau, aTau))
            
            if tau == T - 1:
                episodeRewards.append(sum(rewards[1:]))
                break 
            t+=1

    print(np.mean(episodeRewards[-100:]))

getFinalAverage(0.9,0.2,0.3,2)
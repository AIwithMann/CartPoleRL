"""
CartPole (Gymnasium) balancing with tile coding + semi-gradient TD(0) critic
Actor-Critic architecture:
 - Critic: linear V(s; w) updated with semi-gradient TD(0)
 - Actor : linear softmax policy pi(a|s; theta) updated with TD error (compatible)
Tile coding implements multiple tilings hashed to feature indices.
Designed to run on GPU if available.
"""

import math
import random
import time
from collections import deque, defaultdict

import gymnasium as gym
import numpy as np
import torch

# -------------------------
# Configuration / Hyperparams
# -------------------------
ENV_NAME = "CartPole-v1"
SEED = 1234
NUM_EPISODES = 2000
MAX_EPISODE_STEPS = 500
GAMMA = 0.99
ALPHA_CRITIC = 1e-2      # step size for critic (per-feature)
ALPHA_ACTOR = 1e-3       # step size for actor
NUM_TILINGS = 32
TILE_WIDTH = [0.25, 0.25, 0.01, 0.1]  # approximate width per dimension
HASH_SIZE = 2**18         # total number of possible features (keeps memory bounded)
EVAL_EVERY = 100
EVAL_EPISODES = 10
PRINT_EVERY = 10

# -------------------------
# Device and seeds
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Create environment
# -------------------------
env = gym.make(ENV_NAME)
env.reset(seed=SEED)
n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# -------------------------
# Tile coder implementation
# -------------------------
class TileCoder:
    """
    Simple tile coder producing a sparse set of active feature indices.
    Uses hashing into a fixed-size table (mod hash_size). Works for continuous state.
    """
    def __init__(self, low, high, num_tilings=NUM_TILINGS, tile_width=None, hash_size=HASH_SIZE, seed=SEED):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.dim = len(self.low)
        self.num_tilings = num_tilings
        self.hash_size = hash_size
        self.rng = np.random.RandomState(seed)
        if tile_width is None:
            # default: equal partitioning into ~8 bins per tiling per dim
            self.tile_width = (self.high - self.low) / 8.0
        else:
            self.tile_width = np.array(tile_width, dtype=np.float32)
        # offsets: evenly distribute tilings by small offsets
        self.offsets = np.array([ (i / self.num_tilings) * self.tile_width for i in range(self.num_tilings) ], dtype=np.float32)

    def get_tiles(self, x):
        """
        Return a list of active feature indices (length = num_tilings).
        x: 1D numpy array of state
        """
        x = np.clip(x, self.low, self.high)
        active = []
        for t in range(self.num_tilings):
            # compute tile coordinates per dimension
            coords = np.floor((x + self.offsets[t] - self.low) / self.tile_width).astype(int)
            # combine coords to a single integer via hashing
            # Use a simple randomized hashing: multiply coords by random large int vector and sum
            # to spread across hash_size.
            # Also include tiling index to differentiate tilings.
            # deterministic hashing using rng permutation based on coords and tiling
            h = 0
            for d, c in enumerate(coords):
                # mix coordinate with tiling and dimension
                h ^= (c + 1619 * d + 1013 * t) & 0xffffffff
                # mix bits
                h = (h * 0x27d4eb2d) & 0xffffffff
            idx = int(h % self.hash_size)
            active.append(idx)
        return active

# -------------------------
# Observation ranges for CartPole
# Clip values to reasonable ranges observed commonly
# -------------------------
# Following typical ranges for CartPole
obs_low = np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32)
obs_high = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)

tile_coder = TileCoder(low=obs_low, high=obs_high, num_tilings=NUM_TILINGS, tile_width=TILE_WIDTH, hash_size=HASH_SIZE)

# -------------------------
# Parameters (on device)
# -------------------------
n_features = HASH_SIZE
# Critic weights: V(s) = w^T phi(s)
w = torch.zeros(n_features, dtype=torch.float32, device=device)
# Actor weights: one weight vector per action. preferences = theta[a] dot phi(s)
theta = torch.zeros((n_actions, n_features), dtype=torch.float32, device=device)

# Learning rates converted to per-feature scale
alpha_c = ALPHA_CRITIC / NUM_TILINGS
alpha_a = ALPHA_ACTOR / NUM_TILINGS

# -------------------------
# Utilities
# -------------------------
def features_from_obs(obs):
    """
    obs: numpy array
    returns: list of integer indices (active features)
    """
    return tile_coder.get_tiles(obs)

def value_from_features(indices):
    """
    Evaluate V(s) using sparse active indices.
    """
    if not indices:
        return torch.tensor(0.0, device=device)
    # gather weights and sum
    return w[indices].sum()

def policy_and_logprob(indices):
    """
    Compute softmax policy over actions from linear preferences.
    Returns (probs_tensor_on_device, logprobabilities_tensor)
    """
    # preferences: shape (n_actions,)
    # compute theta[:, indices] dot 1 (since binary features) => sum over indices
    prefs = theta[:, indices].sum(dim=1)  # shape (n_actions,)
    # numerical stability
    maxp = prefs.max()
    exps = torch.exp(prefs - maxp)
    probs = exps / exps.sum()
    logprobs = torch.log(probs + 1e-12)
    return probs, logprobs

def select_action(indices, greedy=False):
    probs, logprobs = policy_and_logprob(indices)
    if greedy:
        action = int(torch.argmax(probs).item())
    else:
        action = int(torch.multinomial(probs, num_samples=1).item())
    return action, probs, logprobs

# -------------------------
# Training Loop (Actor-Critic using semi-gradient TD(0))
# -------------------------
episode_rewards = []
start_time = time.time()

for ep in range(1, NUM_EPISODES + 1):
    obs, info = env.reset(seed=SEED + ep)
    total_r = 0.0
    for t in range(MAX_EPISODE_STEPS):
        indices = features_from_obs(obs)
        # convert indices to tensor indices for gathering
        indices_t = torch.tensor(indices, dtype=torch.long, device=device) if indices else torch.tensor([], dtype=torch.long, device=device)
        # select action
        action, probs, logprobs = select_action(indices_t)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_r += reward

        # compute TD error: delta = r + gamma * V(s') - V(s)
        # V(s)
        v_s = value_from_features(indices_t)
        # V(s')
        next_indices = features_from_obs(next_obs)
        next_indices_t = torch.tensor(next_indices, dtype=torch.long, device=device) if next_indices else torch.tensor([], dtype=torch.long, device=device)
        v_sp = value_from_features(next_indices_t) if not done else torch.tensor(0.0, device=device)

        delta = torch.tensor(reward, dtype=torch.float32, device=device) + GAMMA * v_sp - v_s

        # Critic update (semi-gradient TD(0)): w <- w + alpha_c * delta * phi(s)
        if indices:
            # w[indices] += alpha_c * delta  (multiple indices may collide; we add to each active)
            w.index_add_(0, indices_t, (alpha_c * delta).expand(len(indices)))

        # Actor update: compatible linear softmax
        # gradient log pi(a|s) = phi(s) * (1 - pi(a)) for chosen action and -phi(s)*pi(b) for others
        if indices:
            # compute probs on device
            probs = probs.to(device)
            # update theta: theta[a] += alpha_a * delta * (1 - pi(a)) * phi
            # and theta[other] += -alpha_a * delta * pi(other) * phi
            # Implementation: for all actions, theta[action, indices] += alpha_a * delta * (I(a==action) - pi[action]) * 1
            # vectorized:
            advantage = (alpha_a * delta).item()
            # create a vector (n_actions,) of coefficients = (I[a] - pi[a])
            coeffs = -probs.clone()
            chosen = action
            coeffs[chosen] += 1.0
            # Now update theta[:, indices] by adding coeffs[:,None] * advantage
            # theta[:, indices] shape (n_actions, n_active)
            # We will construct an outer product coeffs[:,None] * ones(1, n_active)
            # Do in-place updates to avoid allocations where possible:
            # For small num_actions this is fine
            for a in range(n_actions):
                theta[a].index_add_(0, indices_t, coeffs[a].expand(len(indices)) * advantage * torch.ones(len(indices), device=device))

        obs = next_obs
        if done:
            break

    episode_rewards.append(total_r)

    # Logging / evaluation
    if ep % PRINT_EVERY == 0:
        avg_last100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 1 else 0.0
        print(f"Ep {ep:4d} | R: {total_r:6.1f} | Avg100: {avg_last100:6.2f} | t: {t+1:3d}")

    if ep % EVAL_EVERY == 0:
        # Quick evaluation with greedy policy
        eval_rs = []
        for _ in range(EVAL_EPISODES):
            obs, info = env.reset()
            rsum = 0.0
            for _step in range(MAX_EPISODE_STEPS):
                idxs = features_from_obs(obs)
                idxs_t = torch.tensor(idxs, dtype=torch.long, device=device) if idxs else torch.tensor([], dtype=torch.long, device=device)
                a, _, _ = select_action(idxs_t, greedy=True)
                obs, r, term, trunc, info = env.step(a)
                rsum += r
                if term or trunc:
                    break
            eval_rs.append(rsum)
        print(f"--> Eval over {EVAL_EPISODES} episodes: mean return {np.mean(eval_rs):.2f}")

# -------------------------
# Final evaluation and save
# -------------------------
total_time = time.time() - start_time
print(f"Training finished. Episodes: {NUM_EPISODES} Time: {total_time:.1f}s")
# Save weights to disk (cpu)
save = {
    "theta": theta.detach().cpu().numpy(),
    "w": w.detach().cpu().numpy(),
    "tilecoder": {
        "low": tile_coder.low.tolist(),
        "high": tile_coder.high.tolist(),
        "num_tilings": tile_coder.num_tilings,
        "tile_width": tile_coder.tile_width.tolist(),
        "hash_size": tile_coder.hash_size,
    },
}
import pickle
with open("cartpole_ac_tilecoder_td0.pickle", "wb") as f:
    pickle.dump(save, f)
print("Model saved to cartpole_ac_tilecoder_td0.pickle")

# Optional: quick playback rendering (local display). Disabled by default.
def play_greedy(n_episodes=3, render=False):
    _env = gym.make(ENV_NAME, render_mode="human" if render else None)
    for i in range(n_episodes):
        obs, info = _env.reset()
        done = False
        rsum = 0.0
        while not done:
            idxs = features_from_obs(obs)
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device) if idxs else torch.tensor([], dtype=torch.long, device=device)
            a, _, _ = select_action(idxs_t, greedy=True)
            obs, r, term, trunc, info = _env.step(a)
            rsum += r
            done = term or trunc
            if render:
                time.sleep(0.02)
        print(f"Play ep {i+1} return {rsum}")
    _env.close()

# Example: play_greedy(render=False)
# play_greedy(2, render=False)

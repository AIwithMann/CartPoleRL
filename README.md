# CartPole RL: Semi-Gradient SARSA with Tile Coding

A comprehensive implementation of reinforcement learning algorithms on the CartPole-v1 environment, exploring both 1-step and n-step Semi-Gradient SARSA with tile coding feature representation.

## Overview

This repository implements function approximation techniques for the CartPole control problem using:
- **Tile Coding**: Efficient feature representation for continuous state spaces
- **Semi-Gradient SARSA**: On-policy TD learning with linear function approximation
- **Hyperparameter Tuning**: Systematic evaluation of learning parameters across 135+ configurations

## Project Structure

```
CartPoleRL/
├── tilecoding.py          # Tile coding implementation
├── sarsa.py               # 1-step Semi-Gradient SARSA
├── sarsaN.py                  # n-step Semi-Gradient SARSA
├── evaluate(1-step).py            # Hyperparameter tuning for 1-step
├── evaluate(n-step).py  # Hyperparameter tuning for n-step
├── evaluation(1-step).md          # Analysis of 1-step results
├── evaluation(n-step).md     # Analysis of n-step results
└── README.md              # This file
```

## Files Explained

### `tilecoding.py`
Implements the tile coding algorithm for feature representation.

**Key Features:**
- Converts continuous state vectors into discrete tile indices
- Supports multiple overlapping tilings for richer feature representation
- Uses hashing to map high-dimensional tile coordinates to feature indices

**Usage:**
```python
from tilecoding import TileCoding

low = np.array([-4.8, -5.0, -0.418, -5.0])
high = np.array([4.8, 5.0, 0.418, 5.0])
tc = TileCoding(num_tiles=10, num_tilings=4, low=low, high=high, N=8192)

state = np.array([1.5, 0.3, -0.1, 0.8])
features = tc.tileIndices(state)  # Returns list of active feature indices
```

### `sarsa.py`
1-step Semi-Gradient SARSA implementation.

**Algorithm:**
- On-policy temporal difference learning
- Single-step bootstrapping: `TD = R + γ*Q(S', A') - Q(S, A)`
- Linear function approximation with sparse features
- ε-greedy exploration with epsilon decay

**Best Configuration:**
- Gamma: 0.99
- Alpha: 0.1
- Epsilon: 0.3
- Performance: ~419 average reward

### `sarsaN.py`
n-step Semi-Gradient SARSA for multi-step lookahead.

**Algorithm:**
- Collects n rewards before bootstrapping: `G = Σ γ^i * R_i + γ^n * Q(S_n, A_n)`
- Balances bias-variance tradeoff through step selection
- Implemented using tabular return accumulation

**Best Configuration:**
- Gamma: 0.99
- Alpha: 0.1
- Epsilon: 0.2
- N-Steps: 2
- Performance: **489.62 average reward** (97.9% of CartPole max)

### `evaluate.py`
Grid search over hyperparameters for 1-step SARSA.

**Tested Configurations:**
- Gamma: {0.99, 0.95, 0.9}
- Alpha: {0.5, 0.2, 0.1}
- Epsilon: {0.3, 0.2, 0.1}
- Total: 27 configurations

**Output:** Bar charts and analysis of hyperparameter effects

### `evaluation(n-step).py`
Comprehensive hyperparameter tuning for n-step SARSA.

**Tested Configurations:**
- Gamma: {0.99, 0.95, 0.9}
- Alpha: {0.5, 0.2, 0.1}
- Epsilon: {0.3, 0.2, 0.1}
- N-Steps: {1, 2, 3, 5, 10}
- Total: 135 configurations

**Output:** Plots showing individual parameter effects and n-step performance comparison

## Installation & Usage

### Requirements
```bash
pip install gymnasium numpy matplotlib
```

### Running 1-Step SARSA
```bash
python sarsa.py
```

### Running n-Step SARSA
```bash
python sarsaN.py
```

### Hyperparameter Tuning (1-Step)
```bash
python "evaluate(1-step).py"
# Generates: hyperparameter_results.png
```

### Hyperparameter Tuning (n-Step)
```bash
python "evaluation(n-step).py"
# Generates: hyperparameter_results_n_step.png
```

## Detailed Analysis

### 1-Step Results
See `evaluation.md` for comprehensive analysis of:
- Effect of Gamma (discount factor)
- Effect of Alpha (learning rate)
- Effect of Epsilon (exploration rate)
- Configuration comparisons

### n-Step Results
See `analysis_n_step.md` for deep dive into:
- Effect of N on bias-variance tradeoff
- Interaction between N and other hyperparameters
- Why N=2 outperforms both 1-step and higher n-steps
- Learning rate requirements for multi-step methods

## Architecture Details

### Tile Coding
- **Tiles per dimension**: 10
- **Number of tilings**: 4 (offset grids)
- **Hash table size**: 8192
- **State dimensions**: 4 (CartPole observation space)

### Function Approximation
- **Feature type**: Sparse binary (active features set to 1)
- **Number of active features**: ~4 per state
- **Weight vector size**: 16,384 (8192 × 2 actions)
- **Update rule**: Semi-gradient with TD error

### Exploration
- **Policy**: ε-greedy with linear decay
- **Initial epsilon**: 0.3 (or 0.2 for n-step)
- **Final epsilon**: 0.01
- **Decay schedule**: Linear over 1000 episodes

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Tile Coding: Feature Representation for Function Approximation
- CartPole-v1: OpenAI Gymnasium Environment

## Author

Developed for learning reinforcement learning fundamentals through hands-on implementation.

## License

MIT

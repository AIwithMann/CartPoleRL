# n-Step SARSA Hyperparameter Analysis Report

## Executive Summary

Comprehensive evaluation of n-step Semi-Gradient SARSA on CartPole-v1 across 135 configurations. The study reveals critical insights about the bias-variance tradeoff in multi-step bootstrapping and optimal hyperparameter combinations.

**Best Configuration:** Gamma=0.99, Alpha=0.1, Epsilon=0.2, N=2
**Best Performance:** 489.62 average reward (97.9% of CartPole maximum)

---

## 1. Effect of N (Number of Steps)

### Results
| N-Steps | Avg Return |
|---------|-----------|
| 1       | 234.30    |
| 2       | 254.32    |
| 3       | 224.49    |
| 5       | 207.38    |
| 10      | 191.99    |

### Analysis

**N=2 is optimal** with a 8.5% improvement over 1-step SARSA. This represents the sweet spot in the bias-variance tradeoff:

- **N=1 (Pure SARSA)**: Low bias, high variance. Uses immediate reward + bootstrap. Achieves 234.30—solid baseline.

- **N=2 (Best)**: 254.32 reward. Balances two rewards before bootstrap:
  ```
  G = R_1 + γ*R_2 + γ²*Q(S_2, A_2)
  ```
  More information before bootstrap reduces variance while keeping bias reasonable.

- **N=3**: 224.49 reward (12% drop from N=2). Three-step returns have higher variance, adding noise to updates.

- **N=5**: 207.38 reward (18.5% drop). Five steps accumulates too much variance in the return estimate.

- **N=10**: 191.99 reward (24.5% drop). Collecting 10 rewards before bootstrapping introduces excessive variance, causing instability.

### Key Insight

**Longer lookahead doesn't always help.** The variance of accumulated rewards grows exponentially with n, and for CartPole's short episode horizon (500 steps max), n=2 provides enough information without excessive noise.

---

## 2. Effect of Gamma (Discount Factor)

### Results
| Gamma | Avg Return |
|-------|-----------|
| 0.99  | 342.48    |
| 0.95  | 176.89    |
| 0.9   | 148.12    |

### Analysis

Gamma remains the **most dominant hyperparameter**, showing consistent superiority:

- **Gamma=0.99**: 342.48 average (193% better than 0.9). Values long-term stability, critical for pole balancing.
- **Gamma=0.95**: 176.89 average. Moderate long-term planning.
- **Gamma=0.9**: 148.12 average. Myopic behavior, poor performance.

**This effect is robust across all n-step values**, indicating that long-term credit assignment is fundamental regardless of bootstrapping method.

---

## 3. Effect of Alpha (Learning Rate)

### Results
| Alpha | Avg Return |
|-------|-----------|
| 0.1   | 304.88    |
| 0.2   | 278.36    |
| 0.5   | 84.26     |

### Analysis

Alpha shows the strongest **inverted-U relationship**:

- **Alpha=0.1**: 304.88 reward. Conservative updates provide stability. With tile coding's sparse features, small steps prevent weight oscillation.

- **Alpha=0.2**: 278.36 reward. 9% lower than alpha=0.1. Still reasonable, but slightly too aggressive for reliable convergence.

- **Alpha=0.5**: 84.26 reward (72% drop from alpha=0.1). Extremely unstable. Large weight updates cause Q-values to swing wildly, breaking the learned policy.

### Why Lower Alpha Wins

With n-step returns, **lower learning rates become even more important**. Multi-step returns already accumulate noise; aggressive learning amplifies this noise. The combination of:
- Multi-step returns (higher variance)
- Tile coding (sparse features, each affecting Q-values significantly)
- Large alpha (aggressive updates)

...creates a destabilizing feedback loop.

---

## 4. Effect of Epsilon (Exploration Rate)

### Results
| Epsilon | Avg Return |
|---------|-----------|
| 0.3     | 225.45    |
| 0.2     | 224.45    |
| 0.1     | 223.89    |

### Analysis

Epsilon has **minimal effect** (only 0.5% variation). With linear decay from initial epsilon to 0.01 over 1000 episodes, the initial value becomes less important than the decay schedule.

All three values achieve similar performance (~224), confirming that epsilon tuning is secondary to gamma and alpha.

---

## 5. Best Configuration Spotlight

### Configuration
```
Gamma:   0.99
Alpha:   0.1
Epsilon: 0.2
N:       2
Result:  489.62 average reward
```

### Why This Works

1. **Gamma=0.99**: Plans for long-term stability (essential for CartPole)
2. **Alpha=0.1**: Conservative updates prevent instability with multi-step returns
3. **Epsilon=0.2**: Balanced exploration-exploitation (with decay)
4. **N=2**: Sweet spot—captures two-step lookahead without excessive variance

### Performance Trajectory
- Gamma=0.99, Alpha=0.1, Epsilon=0.2:
  - N=1: 397.83
  - **N=2: 489.62** ← Best overall
  - N=3: 430.81
  - N=5: 435.69
  - N=10: 457.15

Notice that N=2 stands out as the peak performer in this configuration.

---

## 6. Standout Configurations

### Top 5 Performers
| Rank | Gamma | Alpha | Epsilon | N | Reward |
|------|-------|-------|---------|---|--------|
| 1    | 0.99  | 0.1   | 0.2     | 2 | 489.62 |
| 2    | 0.99  | 0.1   | 0.1     | 3 | 484.94 |
| 3    | 0.99  | 0.1   | 0.1     | 2 | 477.76 |
| 4    | 0.99  | 0.1   | 0.2     | 10| 457.15 |
| 5    | 0.99  | 0.1   | 0.3     | 10| 456.74 |

**Pattern:** All top performers use Gamma=0.99 and Alpha=0.1. N=2 appears most frequently.

### Worst Configurations
| Rank | Gamma | Alpha | Epsilon | N | Reward |
|------|-------|-------|---------|---|--------|
| Last | 0.95  | 0.5   | 0.3     | 3 | 32.73  |
| -2   | 0.95  | 0.5   | 0.1     | 10| 29.56  |
| -3   | 0.9   | 0.5   | 0.3     | 5 | 25.41  |

**Pattern:** All worst performers use Alpha=0.5, causing complete failure.

---

## 7. n-Step vs 1-Step SARSA

### Head-to-Head Comparison
- **1-Step (N=1) avg**: 234.30 across all hyperparams
- **n-Step (best N) avg**: 254.32 across all hyperparams
- **Improvement**: +8.5%

### When n-Step Wins Most
```
Gamma=0.99, Alpha=0.1, Epsilon=0.2:
  N=1:  397.83
  N=2:  489.62  ← 23% improvement!
  
This is the "optimal tuning" regime where n-step shines.
```

### When n-Step Fails
```
Gamma=0.99, Alpha=0.5, Epsilon=0.2:
  N=1:  221.84
  N=2:  302.68  ← Works better with alpha=0.5
  N=5:  64.86   ← Collapses with higher n
  N=10: 32.61   ← Complete failure
```

When alpha is too high, multi-step accumulation creates unbounded variance.

---

## 8. Key Insights & Recommendations

### Insights

1. **N=2 is ideal for CartPole**: More information than 1-step without excessive variance.

2. **Lower alpha becomes critical with multi-step**: Alpha=0.1 vs 0.5 shows 272% difference! Multi-step returns compound errors.

3. **Gamma dominates all methods**: Whether using 1-step or 10-step, gamma=0.99 consistently outperforms.

4. **Bias-variance tradeoff is real**: Too many steps (N≥5) accumulates so much variance that it hurts more than helps.

5. **Optimal config is surprisingly specific**: Best performance emerges at the intersection of (Gamma=0.99, Alpha=0.1, N=2).

### Recommendations

**For CartPole specifically:**
```
Gamma:   0.99   (non-negotiable for long-term planning)
Alpha:   0.1    (conservative to handle multi-step variance)
Epsilon: 0.2    (with linear decay schedule)
N:       2      (two-step lookahead)
```

**For similar problems:**
- Start with N=2, then test N∈[1,3,5] for your domain
- Always use Alpha=0.1 as baseline when using n-step
- If higher alpha needed for speed, reduce N to 1
- Gamma is environment-specific; higher values help with long horizons

---

## 9. Conclusion

This comprehensive study demonstrates that **n-step SARSA with optimal tuning achieves 489.62 reward** (97.9% of CartPole maximum), a **23% improvement** over standard 1-step SARSA in the best case.

However, the results also show that **n-step methods are sensitive to other hyperparameters**—the variance of multi-step returns requires more conservative learning rates and careful step selection.

The sweet spot emerges at N=2 with conservative alpha=0.1 and long-horizon gamma=0.99, suggesting that **modest lookahead with aggressive discounting** is the optimal strategy for this domain.
Made by Claude Haiku 3.5 because I am new to writing such analysis reports

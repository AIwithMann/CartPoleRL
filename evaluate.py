import matplotlib.pyplot as plt
import numpy as np
from sarsa import getFinalAverage

gamma = {1: 0.99, 2: 0.95, 3: 0.9}
alpha = {1: 0.5, 2: 0.2, 3: 0.1}
epsilon = {1: 0.3, 2: 0.2, 3: 0.1}

results = {}
inputs = []

for i in gamma:
    for j in alpha:
        for k in epsilon:
            inputs.append((i, j, k))
            results[(i, j, k)] = getFinalAverage(gamma[i], alpha[j], epsilon[k])

averages = [results[x] for x in inputs]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Hyperparameter Effects', fontsize=14)

# All configurations
ax1 = axes[0, 0]
labels = [f"G{a}A{b}E{c}" for a, b, c in inputs]
ax1.bar(range(len(averages)), averages, color='steelblue', edgecolor='black')
ax1.set_xticks(range(len(averages)))
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.set_title('All Configurations')
ax1.set_ylabel('Average Return')
ax1.grid(axis='y', alpha=0.3)

# Gamma
ax2 = axes[0, 1]
gamma_avg = {gamma[i]: np.mean([results[(i, j, k)] for j in alpha for k in epsilon]) for i in gamma}
ax2.plot(list(gamma_avg.keys()), list(gamma_avg.values()), marker='o')
ax2.set_title('Effect of Gamma')
ax2.set_xlabel('Gamma')
ax2.set_ylabel('Avg Return')
ax2.grid(True, alpha=0.3)

# Alpha
ax3 = axes[1, 0]
alpha_avg = {alpha[j]: np.mean([results[(i, j, k)] for i in gamma for k in epsilon]) for j in alpha}
ax3.plot(list(alpha_avg.keys()), list(alpha_avg.values()), marker='s', color='green')
ax3.set_title('Effect of Alpha')
ax3.set_xlabel('Alpha')
ax3.set_ylabel('Avg Return')
ax3.grid(True, alpha=0.3)

# Epsilon
ax4 = axes[1, 1]
epsilon_avg = {epsilon[k]: np.mean([results[(i, j, k)] for i in gamma for j in alpha]) for k in epsilon}
ax4.plot(list(epsilon_avg.keys()), list(epsilon_avg.values()), marker='^', color='orange')
ax4.set_title('Effect of Epsilon')
ax4.set_xlabel('Epsilon')
ax4.set_ylabel('Avg Return')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hyperparameter_results.png")

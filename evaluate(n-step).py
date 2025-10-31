import matplotlib.pyplot as plt
import numpy as np
from sarsaN import getFinalAverage

gamma = {1: 0.99, 2: 0.95, 3: 0.9}
alpha = {1: 0.5, 2: 0.2, 3: 0.1}
epsilon = {1: 0.3, 2: 0.2, 3: 0.1}
n_steps = {1: 1, 2: 2, 3: 3, 4: 5, 5: 10}

results = {}
inputs = []

print("Running hyperparameter grid search...")
for i in gamma:
    for j in alpha:
        for k in epsilon:
            for l in n_steps:
                inputs.append((i, j, k, l))
                avg = getFinalAverage(gamma[i], alpha[j], epsilon[k], n_steps[l])
                results[(i, j, k, l)] = avg
                print(f"Gamma={gamma[i]}, Alpha={alpha[j]}, Epsilon={epsilon[k]}, N={n_steps[l]}: {avg:.2f}")

print("Grid search complete!")
averages = [results[x] for x in inputs]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('n-Step SARSA Hyperparameter Tuning', fontsize=16, fontweight='bold')

# 1. All configurations
ax1 = axes[0, 0]
labels = [f"G{a}A{b}E{c}N{d}" for a, b, c, d in inputs]
bars = ax1.bar(range(len(averages)), averages, color='steelblue', edgecolor='black')
ax1.set_xticks(range(len(averages)))
ax1.set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
ax1.set_title('All Configurations', fontweight='bold')
ax1.set_ylabel('Average Return')
ax1.grid(axis='y', alpha=0.3)

# 2. Effect of Gamma
ax2 = axes[0, 1]
gamma_avg = {}
for i in gamma:
    vals = [results[(i, j, k, l)] for j in alpha for k in epsilon for l in n_steps]
    gamma_avg[gamma[i]] = np.mean(vals)

ax2.plot(list(gamma_avg.keys()), list(gamma_avg.values()), marker='o', linewidth=2, markersize=8, color='coral')
ax2.set_title('Effect of Gamma', fontweight='bold')
ax2.set_xlabel('Gamma')
ax2.set_ylabel('Avg Return')
ax2.grid(True, alpha=0.3)
for g, avg in zip(gamma_avg.keys(), gamma_avg.values()):
    ax2.text(g, avg, f'{avg:.0f}', ha='center', va='bottom')

# 3. Effect of Alpha
ax3 = axes[1, 0]
alpha_avg = {}
for j in alpha:
    vals = [results[(i, j, k, l)] for i in gamma for k in epsilon for l in n_steps]
    alpha_avg[alpha[j]] = np.mean(vals)

ax3.plot(list(alpha_avg.keys()), list(alpha_avg.values()), marker='s', linewidth=2, markersize=8, color='lightgreen')
ax3.set_title('Effect of Alpha', fontweight='bold')
ax3.set_xlabel('Alpha')
ax3.set_ylabel('Avg Return')
ax3.grid(True, alpha=0.3)
for a, avg in zip(alpha_avg.keys(), alpha_avg.values()):
    ax3.text(a, avg, f'{avg:.0f}', ha='center', va='bottom')

# 4. Effect of N (n-steps)
ax4 = axes[1, 1]
n_avg = {}
for l in n_steps:
    vals = [results[(i, j, k, l)] for i in gamma for j in alpha for k in epsilon]
    n_avg[n_steps[l]] = np.mean(vals)

ax4.plot(list(n_avg.keys()), list(n_avg.values()), marker='^', linewidth=2, markersize=8, color='skyblue')
ax4.set_title('Effect of N (Number of Steps)', fontweight='bold')
ax4.set_xlabel('N')
ax4.set_ylabel('Avg Return')
ax4.grid(True, alpha=0.3)
for n, avg in zip(n_avg.keys(), n_avg.values()):
    ax4.text(n, avg, f'{avg:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("hyperparameter_results_n_step.png", dpi=300, bbox_inches='tight')
print("Plot saved as hyperparameter_results_n_step.png")
plt.show()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nBest configuration:")
best_config = max(results, key=results.get)
best_value = results[best_config]
print(f"  Gamma={gamma[best_config[0]]}, Alpha={alpha[best_config[1]]}, Epsilon={epsilon[best_config[2]]}, N={n_steps[best_config[3]]}")
print(f"  Average Return: {best_value:.2f}")

print(f"\nHyperparameter averages:")
print(f"  Gamma: {gamma_avg}")
print(f"  Alpha: {alpha_avg}")
print(f"  N-Steps: {n_avg}")
print("="*60)

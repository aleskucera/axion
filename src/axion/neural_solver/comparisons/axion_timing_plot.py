import numpy as np
import matplotlib.pyplot as plt

#                                    mean       std
control = np.array([0.013548, 0.000475])
initial_guess = np.array([0.056488, 0.000753])
newton_linearization = np.array([0.062619, 0.000750])
newton_linear_solve = np.array([0.492124, 0.026524])
#newton_linesearch = np.array([0.592459, 0.020963])
all = np.stack(
    [
        control,
        initial_guess,
        newton_linearization,
        newton_linear_solve,
    ],
    axis=0,
)

means = all[:, 0]
stds = all[:, 1]

labels = [
    "control calculation",
    "newton: initial_guess",
    "newton: system linearization",
    "newton: linear solve",
]
x = np.arange(len(labels))

fig, ax = plt.subplots()
ax.set_axisbelow(True)
ax.bar(x, means, yerr=stds, capsize=4, ecolor="black", color="steelblue", edgecolor="black")
ax.set_xticks(x, labels, rotation=45, ha="right")
ax.set_ylabel("Time [ms]")
ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
ax.set_title("Axion Simulator Timing experiments")
fig.tight_layout()
plt.show()

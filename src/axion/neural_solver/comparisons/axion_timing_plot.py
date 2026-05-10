import numpy as np
import matplotlib.pyplot as plt

tick_fontsize = 13
legend_fontsize = 13

labels = [
    "Collision\n detection",
    "Contact\n preprocessing",
    "Newton's method\ninitial guess",
    "Newton system\nsolving",
    "Best iterate\nbacktracking",
    "Output\ncopying",
]
means = np.array([0.350, 1.272, 0.015, 8.734, 0.026, 0.020])
p95s = np.array([0.369, 1.598, 0.014, 13.877, 0.023, 0.014])

x = np.arange(len(labels))
width = 0.4

fig, ax = plt.subplots()
ax.set_axisbelow(True)
ax.bar(x - width / 2, means, width, label="mean", color="steelblue", edgecolor="black")
ax.bar(x + width / 2, p95s, width, label="p95", color="indianred", edgecolor="black")
ax.set_xticks(x, labels, rotation=0, ha="center", fontsize = tick_fontsize)
ax.set_ylabel("Time [ms]", fontsize=tick_fontsize)
ax.tick_params(axis="y", labelsize=tick_fontsize)
ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
ax.set_title("Axion simulator timing experiments (end to end)")
ax.legend(fontsize=legend_fontsize)
fig.tight_layout()
plt.show()

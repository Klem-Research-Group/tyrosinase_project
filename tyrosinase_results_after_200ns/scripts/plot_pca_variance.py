import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User Input
# -----------------------------

input_file = "plot_pca_variance.dat"
output_png = "pca_variance_plot.png"

# -----------------------------
# Read the input file
# -----------------------------

df = pd.read_csv(input_file, sep=None, engine='python', header=None)
df.columns = ["PC", "Variance", "Cumulative"]

# -----------------------------
# Plot
# -----------------------------

fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for variance
ax1.bar(
    df["PC"],
    df["Variance"],
    label="Variance",
    alpha=0.7
)

ax1.set_xlabel("PCs", fontsize=18)
ax1.set_ylabel("Variance", fontsize=18)
ax1.set_ylim(0, 0.4)

# Tick label fontsize
ax1.tick_params(axis='both', labelsize=16)

# Line plot for cumulative variance
ax2 = ax1.twinx()

ax2.plot(
    df["PC"],
    df["Cumulative"],
    marker="o",
    linewidth=2,
    label="Cumulative Variance"
)

ax2.set_ylabel("Cumulative Variance", fontsize=18)
ax2.set_ylim(0, 1)

# Tick label fontsize for secondary axis
ax2.tick_params(axis='both', labelsize=16)

# Combined legend
lines, labels = [], []

for ax in [ax1, ax2]:
    l, lab = ax.get_legend_handles_labels()
    lines += l
    labels += lab

ax1.legend(
    lines,
    labels,
    loc="best",
    fontsize=16
)

plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.show()

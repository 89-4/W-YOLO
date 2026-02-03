import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------
# === Replace these with your real data ===
# Order: [mAP50, mAP50-90, GFLOPs, Params(M)]
data = {
    "Baseline":      [0.919, 0.7107, 25.5, 9.5],   # ← original model
    "LAMP":          [0.9031, 0.6743, 15.9, 5.2],
    "L1":            [0.8972, 0.6684, 15.7, 5.9],
    "Group-Taylor":  [0.9067, 0.6808, 15.8, 4.8],
    "Slim":          [0.8931, 0.658, 15.7, 5.9],
    "Group-Hessian":  [0.9047, 0.6790, 15.9, 5.2],
}
# ---------------------------

methods = list(data.keys())

# Original DataFrame
df = pd.DataFrame.from_dict(
    data,
    orient='index',
    columns=["mAP50", "mAP50-90", "GFLOPs", "Params"]
)

# ==============================
# === Option 2: normalize relative to Baseline
# GFLOPs_rel = GFLOPs / Baseline_GFLOPs
# Params_rel = Params / Baseline_Params
# ==============================
baseline = df.loc["Baseline"]

df_rel = df.copy()
df_rel["GFLOPs_rel"] = df["GFLOPs"] / baseline["GFLOPs"]
df_rel["Params_rel"] = df["Params"] / baseline["Params"]

# Data used for plotting
df_plot = df_rel[["mAP50", "mAP50-90", "GFLOPs_rel", "Params_rel"]]
metrics = ["mAP50", "mAP50-90", "GFLOPs / Baseline", "Params / Baseline"]

print("Relative normalized data:\n", df_plot)

# ---------------------------
# === Plotting
x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(9, 5))

for method in methods:
    ax.plot(
        x,
        df_plot.loc[method].values,
        marker='o',
        linewidth=2,
        markersize=6,
        label=method
    )

# x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_xlabel("Metrics (mAP raw values; GFLOPs & Params relative to Baseline)")

# y-axis label
ax.set_ylabel("Value (mAP raw or relative value)")
ax.set_title("Pruning Methods: Trade-off Plot")
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(loc='best')

plt.tight_layout()

# Save figures
fig.savefig("./pic/pruning_method_tradeoff_rel.png", dpi=300)
fig.savefig("./pic/pruning_method_tradeoff_rel.svg")
print("Saved: pruning_tradeoff_rel.png and pruning_tradeoff_rel.svg")

plt.show()

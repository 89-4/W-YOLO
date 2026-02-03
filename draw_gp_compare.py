import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# ==== Replace these with your real data ====
# Structure: {model_name: [mAP50, mAP50-90, GFLOPs, Params(M)]}
results = {
    "Baseline":       [0.919, 0.7107, 25.5, 9.5],
    "GT 1.2×":        [0.917, 0.6974, 19.6, 7.6],
    "GT 1.4×":        [0.9089, 0.6836, 16.5, 5.4],
    "GT 1.6×":        [0.9067, 0.6808, 15.8, 4.8],
    "GT 1.8×":        [0.8934, 0.6605, 14.1, 3.7],
    "GT 2.0×":        [0.8757, 0.6345, 13.0, 2.9],
}
# -----------------------------

# Metric names for original table
metrics = ["mAP50", "mAP50-90", "GFLOPs", "Params (M)"]

# Build DataFrame
df = pd.DataFrame.from_dict(results, orient="index", columns=metrics).astype(float)
print("Original data preview:\n", df)

# Baseline row
baseline = df.loc["Baseline"]

# Create a copy and normalize GFLOPs and Params by baseline values
df_rel = df.copy()
df_rel["GFLOPs"] = df_rel["GFLOPs"] / baseline["GFLOPs"]
df_rel["Params (M)"] = df_rel["Params (M)"] / baseline["Params (M)"]

# Prepare plotting metrics: keep mAPs raw, show normalized names for FLOPs/Params
metrics_plot = ["mAP50", "mAP50-90", "GFLOPs / Baseline", "Params / Baseline"]
df_plot = df_rel.rename(columns={"GFLOPs": "GFLOPs / Baseline", "Params (M)": "Params / Baseline"})[[
    "mAP50", "mAP50-90", "GFLOPs / Baseline", "Params / Baseline"
]]

print("\nNormalized (relative to Baseline) data preview:\n", df_plot)

# Plot
x = np.arange(len(metrics_plot))
plt.figure(figsize=(9, 5))
for model_name, row in df_plot.iterrows():
    y = row.values.astype(float)
    plt.plot(
        x,
        y,
        marker='o',
        linewidth=2,
        markersize=6,
        label=model_name
    )

plt.xticks(x, metrics_plot)
plt.xlabel("Metrics (mAP raw values; GFLOPs & Params relative to Baseline)")
plt.ylabel("Value (mAP raw or relative value)")
plt.title("Group-Taylor Pruning: Trade-off Plot")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='best')

# Save figures (vector + high-res raster)
plt.tight_layout()
plt.savefig("./pic/group_taylor_metrics_rel.svg")
plt.savefig("./pic/group_taylor_metrics_rel.png", dpi=300)

plt.show()
print("Saved: baseline_vs_group_taylor_metrics_relative.svg, baseline_vs_group_taylor_metrics_relative.png")

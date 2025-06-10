import json
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.labelsize"] = 12 
plt.rcParams["ytick.labelsize"] = 12

metrics_path = "fig_scripts/maskrcnn_metrics.json"

# load JSON
with open(metrics_path, 'r') as f:
    lines = f.readlines()
records = [json.loads(line) for line in lines if "iteration" in line] # parse each line as a JSON object
df = pd.DataFrame(records)

# Subplots for total loss and accuracy
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Total loss subplot
axs[0].plot(df["iteration"], df["total_loss"], color='k')
axs[0].set_ylabel("Total Loss", fontsize=14)

# Mask R-CNN accuracy subplot
axs[1].plot(df["iteration"], df["mask_rcnn/accuracy"], color='k')
axs[1].set_ylabel("Mask R-CNN Training Accuracy", fontsize=14)
axs[1].set_xlabel("Iteration", fontsize=14)

plt.rcParams["xtick.labelsize"] = 12  
plt.rcParams["ytick.labelsize"] = 12
plt.tight_layout()
plt.savefig("fig_scripts/figures/maskrcnn_metrics_plot.png")

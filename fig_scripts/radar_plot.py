###  Generate radar plot for paper

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 15,
    "axes.titlesize": 14
})

# Model data is output of segmask_test.py
metrics = ["Video 3", "Video 4", "Video 5", "Video 6", "Video 7"]
model_data = {
    "Per-frame Prompting + SAM2": [0.852, 0.879, 0.669, 0.845, 0.737],
    "HiL-Tracking + SAM2 Video": [0.847, 0.850, 0.838, 0.850, 0.852],
    "FLAIR": [0.855, 0.863, 0.814, 0.717, 0.854]
}

# Number of variables
num_vars = len(metrics)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Set up the radar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Clear default labels
ax.set_xticklabels([])

# Calculate label positions and add custom labels
label_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
for i, angle in enumerate(label_angles):
    x = np.cos(angle)
    y = np.sin(angle)
    
    # Adjust the horizontal alignment and position based on angle
    if -0.1 <= x <= 0.1:
        if y > 0:
            ax.text(angle, 1.15, metrics[i], ha='center', va='bottom', fontsize=18)
        else:
            ax.text(angle, 1.15, metrics[i], ha='center', va='top', fontsize=18)
    elif x < 0:
        ax.text(angle, 1.1, metrics[i], ha='right', va='center', fontsize=18)
    else:
        ax.text(angle, 1.1, metrics[i], ha='left', va='center', fontsize=18)

# Draw y-labels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
plt.ylim(0, 1)

colors = {
    "Per-frame Prompting + SAM2": "#9A348E",
    "HiL-Tracking + SAM2 Video": "#66cdaa",
    "FLAIR": "#F18F01"
}

# Plot each model
for model, values in model_data.items():
    filtered_values = [v if not np.isnan(v) else 0 for v in values]
    filtered_values += filtered_values[:1]
    
    ax.plot(angles, filtered_values, linewidth=2, linestyle='solid', label=model, color=colors[model])
    ax.fill(angles, filtered_values, color=colors[model], alpha=0.1)
    
    for i, (angle, value) in enumerate(zip(angles, filtered_values[:-1])):
        if values[i] != 0 and not np.isnan(values[i]):
            ax.scatter(angle, value, color=colors[model], edgecolor='black', s=50, zorder=3)


labels = [
    "Per-frame \n Prompting \n + SAM2",
    "HiL-Tracking \n + SAM2 Video",
    "FLAIR"
]   
# Create legend with custom handles
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colors[key], lw=5) for key in colors.keys()]
custom_labels = labels
plt.legend(custom_lines, custom_labels, loc='upper right', bbox_to_anchor=(1.3, 1.1))

#plt.tight_layout()
fig.set_size_inches(16, 8) 

plt.savefig("RadarPlot.png", dpi=600, bbox_inches='tight') 

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.titlesize": 14
})

# Values from segmask_test.py
metrics = ["Video 1", "Video 2"]
model_data = {
    "Per-frame \n Prompting \n+ SAM2": [0.756, 0.780],
    "YOLOv8 \n+ SAM2": [0.595, 0.014],
    "DETR \n+ SAM2": [0.700, 0],
    "Mask R-CNN": [0.697, 0.141],
    " HiL-Tracking \n+ SAM2 Video": [0.734, 0.839],
    "FLAIR": [0.719, 0.807]
}

# Parameters for plotting
n_groups = len(model_data)
bar_width = 0.35
index = np.arange(n_groups)

# Convert dictionary to lists for easy access
model_names = list(model_data.keys())
video_1_values = [model_data[model][0] for model in model_names]
video_2_values = [model_data[model][1] for model in model_names]

# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
bar1 = ax.bar(index, video_1_values, bar_width, label="Video 1", color='#26619C')
bar2 = ax.bar(index + bar_width, video_2_values, bar_width, label="Video 2", color='#E7774A')

# Set y-axis limit
ax.set_ylim(0, 1)

# Labeling
ax.set_ylabel("Dice Score")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(model_names)

# Move the legend outside the plot
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Display plot
plt.tight_layout()
plt.savefig("Two_Vids_Barplot.png", dpi=600, bbox_inches='tight')

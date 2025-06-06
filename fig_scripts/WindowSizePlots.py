import numpy as np
import matplotlib.pyplot as plt

# Set font to Arial explicitly
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.titlesize": 16   
})

metrics = ["Video 3", "Video 4", "Video 5", "Video 6", "Video 7"]
window_labels = ["1", "2", "3", "5", r"$N_{\mathrm{frames}}$"]

time = {
    "Video 3": [5.82, 6.53, 6.55, 7.47, 9.22],
    "Video 4": [6.8, 7.23, 7.47, 7.72, 10.03],
    "Video 5": [7.02, 12.12, 15.78, 19.07, 34.83],
    "Video 6": [9.42, 10.4, 12.85, 16.67, 48.77],
    "Video 7": [5.12, 5.67, 6.48, 6.28, 7.9]
}

dice = {
    "Video 3": [0.855, 0.852, 0.855, 0.852, 0.850],
    "Video 4": [0.867, 0.866, 0.863, 0.864, 0.861],
    "Video 5": [0, 0.802, 0.814, 0.831, 0.832],
    "Video 6": [0.706, 0.714, 0.717, 0.717, 0.720],
    "Video 7": [0, 0, 0.854, 0.854, 0.854]
}

# Line styles
linestyles = [
    'solid',
    (0, (5, 5)),
    (0, (3, 5, 1, 5)),
    (0, (10, 7)),
    (0, (1, 8)),
]

# Time vs. Window Size Plot
plt.figure(figsize=(10, 6))
for i, (video, style) in enumerate(zip(metrics, linestyles)):
    total_time = [t for t in time[video]]
    plt.plot(window_labels, total_time, label=video, color='black', linestyle=style)
plt.xlabel("Window Size")
plt.ylabel("Total Time (minutes)")
#plt.legend()
plt.savefig("Total_Time_vs_Window_Size.png", dpi=600, bbox_inches='tight')
plt.close()

# Dice vs. Window Size Plot
plt.figure(figsize=(10, 6))
for video, style in zip(metrics, linestyles):
    plt.plot(window_labels, dice[video], label=video, color='black', linestyle=style)
plt.xlabel("Window Size")
plt.ylabel("Dice Coefficient")
plt.ylim(0, 1)
plt.legend()
plt.savefig("Dice_vs_Window_Size.png", dpi=600, bbox_inches='tight')
plt.close()

# Code for plotting predicted vs actual lengths of sharks (Figure 7)
# creates scatter plot and violin plot. They are combined with the inset images in powerpoint.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.photogrammetry import pixels_to_meters
from utils.stats import percent_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# following numbers are the output from shark_length.py
# pixel lengths of sharks by image
# Load blacktip data
actual_blacktip_pix = np.loadtxt("fig_scripts/length_results/blacktip/gt.txt").tolist()
hil_blacktip_pix    = np.loadtxt("fig_scripts/length_results/blacktip/hil.txt").tolist()
flair_blacktip_pix  = np.loadtxt("fig_scripts/length_results/blacktip/flair.txt").tolist()

# Load white1 data
actual_white1_pix = np.loadtxt("fig_scripts/length_results/white/gt.txt").tolist()
hil_white1_pix    = np.loadtxt("fig_scripts/length_results/white/hil.txt").tolist()
flair_white1_pix  = np.loadtxt("fig_scripts/length_results/white/flair.txt").tolist()

# Load nurse (344) data
actual_344_pix = np.loadtxt("fig_scripts/length_results/nurse/gt.txt").tolist()
hil_344_pix    = np.loadtxt("fig_scripts/length_results/nurse/hil.txt").tolist()
flair_344_pix  = np.loadtxt("fig_scripts/length_results/nurse/flair.txt").tolist()

# convert to meters from pixels
# assume the median blacktip is around 1m
# set drone height to match that, real flight metadata is unknown. Assume constant altitude and nadir angle, which is not true
actual_blacktip = [pixels_to_meters(measurement, altitude=10) for measurement in actual_blacktip_pix]
flair_blacktip = [pixels_to_meters(measurement, altitude=10) for measurement in flair_blacktip_pix]
hil_blacktip = [pixels_to_meters(measurement, altitude=10) for measurement in hil_blacktip_pix]
print(f'median blacktip length: {np.median(actual_blacktip)} m')

# assume the median juvenile whiteshark should be around 3m, following the upper size limit of a juvenile white shark from literature
# set drone height to match that, real flight metadata is unknown
actual_white1 = [pixels_to_meters(measurement, altitude=57) for measurement in actual_white1_pix]
flair_white1 = [pixels_to_meters(measurement, altitude=57) for measurement in flair_white1_pix]
hil_white1 = [pixels_to_meters(measurement, altitude=57) for measurement in hil_white1_pix]
print(f'median white length: {np.median(actual_white1)} m')

actual_344 = [pixels_to_meters(measurement) for measurement in actual_344_pix]
flair_344 = [pixels_to_meters(measurement) for measurement in flair_344_pix]
hil_344 = [pixels_to_meters(measurement) for measurement in hil_344_pix]
print(f'median nurse length: {np.median(actual_344)} m')

data = {
    "Actual - White": actual_white1,
    "FLAIR - White": flair_white1,
    "HiL - White": hil_white1,
    "Actual - Nurse": actual_344,
    "FLAIR - Nurse": flair_344,
    "HiL - Nurse": hil_344,
    "Actual - Blacktip": actual_blacktip,
    "FLAIR - Blacktip": flair_blacktip,
    "HiL - Blacktip": hil_blacktip,
}
species_map = {
    "Actual - White": "White Shark",
    "FLAIR - White": "White Shark",
    "HiL - White": "White Shark",
    "Actual - Nurse": "Nurse Shark",
    "FLAIR - Nurse": "Nurse Shark",
    "HiL - Nurse": "Nurse Shark",
    "Actual - Blacktip": "Blacktip Reef Shark",
    "FLAIR - Blacktip": "Blacktip Reef Shark",
    "HiL - Blacktip": "Blacktip Reef Shark",
}

# define colors
palette = sns.color_palette("deep", n_colors=10)  # get 3 colors from the "deep" palette
blue = palette[0]
green = palette[2]
red = palette[3]
purple = palette[4]
colors = {
    "White Shark": red,  
    "Nurse Shark": purple, 
    "Blacktip Reef Shark": green, 
}

#calculate percent error statistics
flair_percent_error_white = percent_error(actual_white1, flair_white1)
flair_percent_error_nurse = percent_error(actual_344, flair_344)
flair_percent_error_blacktip = percent_error(actual_blacktip, flair_blacktip)
flair_percent_errors = np.concatenate([flair_percent_error_white, flair_percent_error_nurse, flair_percent_error_nurse])

hil_percent_error_white = percent_error(actual_white1, hil_white1)
hil_percent_error_nurse = percent_error(actual_344, hil_344)
hil_percent_error_blacktip = percent_error(actual_blacktip, hil_blacktip)
hil_percent_errors = np.concatenate([hil_percent_error_white, hil_percent_error_nurse, hil_percent_error_nurse])

# calculate and print mean ± SD for each array
print("\n")
print(f"Blacktip (Actual) Mean ± SD = {np.mean(actual_blacktip):.2f} ± {np.std(actual_blacktip):.2f} m")
print(f"Blacktip (FLAIR) Mean ± SD = {np.mean(flair_blacktip):.2f} ± {np.std(flair_blacktip):.2f} m")
print(f"Blacktip (HiL) Mean ± SD = {np.mean(hil_blacktip):.2f} ± {np.std(hil_blacktip):.2f} m")

print(f"White Shark (Actual) Mean ± SD = {np.mean(actual_white1):.2f} ± {np.std(actual_white1):.2f} m")
print(f"White Shark (FLAIR) Mean ± SD = {np.mean(flair_white1):.2f} ± {np.std(flair_white1):.2f} m")
print(f"White Shark (HiL) Mean ± SD = {np.mean(hil_white1):.2f} ± {np.std(hil_white1):.2f} m")

print(f"Nurse Shark (Actual) Mean ± SD = {np.mean(actual_344):.2f} ± {np.std(actual_344):.2f} m")
print(f"Nurse Shark (FLAIR) Mean ± SD = {np.mean(flair_344):.2f} ± {np.std(flair_344):.2f} m")
print(f"Nurse Shark (HiL) Mean ± SD = {np.mean(hil_344):.2f} ± {np.std(hil_344):.2f} m")

# FLAIR percent error and MAE
print("\n")
print("_______________________RESULTS FOR FLAIR:________________________")
print(f'White Shark Mean percent error ± SD: {np.mean(flair_percent_error_white):.2f} ± {np.std(flair_percent_error_white):.2f} %')
print(f'Nurse Shark Mean percent error ± SD: {np.mean(flair_percent_error_nurse):.2f} ± {np.std(flair_percent_error_nurse):.2f} %')
print(f'Blacktip Shark Mean percent error ± SD: {np.mean(flair_percent_error_blacktip):.2f} ± {np.std(flair_percent_error_blacktip):.2f} %')

print("\n")
print(f'White Shark percent error min: {np.min(flair_percent_error_white):.2f} %, max: {np.max(flair_percent_error_white):.2f} %, n = {len(flair_percent_error_white)}')
print(f'Nurse Shark percent error min: {np.min(flair_percent_error_nurse):.2f} %, max: {np.max(flair_percent_error_nurse):.2f} %, n = {len(flair_percent_error_nurse)}')
print(f'Blacktip Shark percent error min: {np.min(flair_percent_error_blacktip):.2f} %, max: {np.max(flair_percent_error_blacktip):.2f} %, n = {len(flair_percent_error_blacktip)}')

print("\n")
print(f'Overall Mean percent error ± SD: {np.mean(flair_percent_errors):.2f} ± {np.std(flair_percent_errors):.2f} %')
print(f'Overall percent error min: {np.min(flair_percent_errors):.2f} %, max: {np.max(flair_percent_errors):.2f} %, n = {flair_percent_errors.shape[0]}')

print('\n')
print(f'White shark mean ± mean absolute error: {np.mean(actual_white1):.2f} ± {mean_absolute_error(actual_white1, flair_white1):.2f} m')
print(f'Nurse shark mean ± mean absolute error: {np.mean(actual_344):.2f} ± {mean_absolute_error(actual_344, flair_344):.2f} m')
print(f'Blacktip shark mean ± mean absolute error: {np.mean(actual_blacktip):.2f} ± {mean_absolute_error(actual_blacktip, flair_blacktip):.2f} m')

# HiL: calculate and print mean ± SD for each array
print("\n")
print("_______________________RESULTS FOR HIL:________________________")
print(f'White Shark Mean percent error ± SD: {np.mean(hil_percent_error_white):.2f} ± {np.std(hil_percent_error_white):.2f} %')
print(f'Nurse Shark Mean percent error ± SD: {np.mean(hil_percent_error_nurse):.2f} ± {np.std(hil_percent_error_nurse):.2f} %')
print(f'Blacktip Shark Mean percent error ± SD: {np.mean(hil_percent_error_blacktip):.2f} ± {np.std(hil_percent_error_blacktip):.2f} %')

print("\n")
print(f'White Shark percent error min: {np.min(hil_percent_error_white):.2f} %, max: {np.max(hil_percent_error_white):.2f} %, n = {len(hil_percent_error_white)}')
print(f'Nurse Shark percent error min: {np.min(hil_percent_error_nurse):.2f} %, max: {np.max(hil_percent_error_nurse):.2f} %, n = {len(hil_percent_error_nurse)}')
print(f'Blacktip Shark percent error min: {np.min(hil_percent_error_blacktip):.2f} %, max: {np.max(hil_percent_error_blacktip):.2f} %, n = {len(hil_percent_error_blacktip)}')

print("\n")
print(f'Overall Mean percent error ± SD: {np.mean(hil_percent_errors):.2f} ± {np.std(hil_percent_errors):.2f} %')
print(f'Overall percent error min: {np.min(hil_percent_errors):.2f} %, max: {np.max(hil_percent_errors):.2f} %, n = {hil_percent_errors.shape[0]}')

print('\n')
print(f'White shark mean ± mean absolute error: {np.mean(actual_white1):.2f} ± {mean_absolute_error(actual_white1, hil_white1):.2f} m')
print(f'Nurse shark mean ± mean absolute error: {np.mean(actual_344):.2f} ± {mean_absolute_error(actual_344, hil_344):.2f} m')
print(f'Blacktip shark mean ± mean absolute error: {np.mean(actual_blacktip):.2f} ± {mean_absolute_error(actual_blacktip, hil_blacktip):.2f} m')


# create a figure with 4 subplots, sharing equal space between scatter plot and violin plots
fig, axs = plt.subplots(1, 4, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 0.2, 0.2, 0.2]})

# --- plot 1: predicted vs actual lengths scatter plot ---
x = np.linspace(0, 8, 100)
axs[0].plot(x, x, color="grey", linestyle="--", linewidth=1, zorder=1)

axs[0].scatter(actual_white1, flair_white1, color=colors["White Shark"], label="White Shark", zorder=2)
axs[0].scatter(actual_344, flair_344, color=colors["Nurse Shark"], label="Nurse Shark", zorder=3)
axs[0].scatter(actual_blacktip, flair_blacktip, color=colors["Blacktip Reef Shark"], label="Blacktip Reef Shark", zorder=2)

axs[0].set_xlim(0, 8)
axs[0].set_ylim(0, 8)
axs[0].set_xlabel("Ground truth length (m)", fontsize=20)
axs[0].set_ylabel("FLAIR predicted length (m)", fontsize=20)


axs[0].legend(title='Species', fontsize=18, title_fontproperties={"weight": "bold", "size":18})
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=16)
axs[0].set_xticklabels(axs[0].get_xticks(), fontsize=16)
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# --- plot 2-4: violin plots for each species ---
# prepare data for violin plots
categories_flat = []
values = []
species = []

for category, vals in data.items():
    species_name = species_map[category]
    categories_flat.extend([category] * len(vals))
    values.extend(vals)
    species.extend([species_name] * len(vals))

df = pd.DataFrame({"Category": categories_flat, "Value": values, "Species": species})

species_list = ["White Shark", "Nurse Shark", "Blacktip Reef Shark"]
for i, species_name in enumerate(species_list):
    sns.violinplot(
        data=df[df["Species"] == species_name],
        x="Category",
        y="Value",
        ax=axs[i + 1],
        palette=[colors[species_name]],
    )
    axs[i + 1].set_xlabel(None)
    if i == 0:
        axs[i + 1].set_ylabel('Length (m)', fontsize=20)
    else:
        axs[i + 1].set_ylabel(None)
    axs[i + 1].set_xticklabels(["GT", "FLAIR", "HiL"], fontsize=14, fontweight='bold')
    axs[i + 1].set_yticklabels(axs[i + 1].get_yticks(), fontsize=16)
    axs[i + 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

axs[1].set_ylim(3, 8.5)
axs[2].set_ylim(1, 2.1)
axs[3].set_ylim(0, 2.2)

plt.tight_layout()
plt.savefig("fig_scripts/figures/biometrics_length_plot.png", dpi=600, bbox_inches="tight")



# # plot 3: boxplot (not used in publication)
# # preserve the order of categories
# categories = list(data.keys())

# # flatten data for plotting
# values = [val for key in data.values() for val in key]
# categories_flat = [cat for cat, vals in data.items() for _ in vals]
# category_colors = {cat: colors[species_map[cat]] for cat in categories}

# # create the boxplot
# plt.figure(figsize=(8, 8))
# sns.boxplot(
#     x=categories_flat,
#     y=values,
#     color="white",  
#     linecolor='black',
#     width=0.6,
#     fliersize=0  
# )

# # add jittered points with custom palette across the entire box width
# sns.stripplot(
#     x=categories_flat,
#     y=values,
#     palette=category_colors,
#     jitter=0.3,  # Wider jitter to cover box width
#     alpha=0.6,
#     zorder=2
# )

# # remove x-ticks and x-labels
# plt.xticks([])  # Remove x-ticks
# plt.xlabel("")  # Remove x-axis label

# # customize the plot
# plt.ylim(0, 8)
# plt.ylabel("Length (m)", fontsize=12)
# plt.tight_layout()

# # plt.savefig("figures/length_boxplot.png", dpi=600, bbox_inches="tight")
# # plt.show()





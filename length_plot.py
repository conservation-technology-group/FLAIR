# Code for plotting predicted vs actual lengths of sharks (Figure 7)
# creates scatter plot and violin plot. They are combined with the inset images in powerpoint.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.photogrammetry import pixels_to_meters
from utils.stats import percent_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter

# following numbers are the output from shark_length.py
# pixel lengths of sharks by image
actual_blacktip_pix = [167.71067811865487, 168.26692300949753, 180.56006556446587, 193.6464730981962, 277.6256035049745, 291.5653114322677, 327.8072741868992, 355.27329066365246, 370.508756990548, 411.21121223894255, 352.85461871046516, 378.1996510392651, 314.1787257270679, 390.72870627774813, 439.3540505454627, 437.8585115498943, 385.771840715679, 343.3062224566841, 405.6024398881978, 504.6601884827532, 477.7101800672151, 438.8704349324174, 403.1108359138474, 501.4577978065342, 501.96485008596244]
pred_blacktip_pix = [182.85915476408223, 199.0530230824658, 197.15569661430675, 212.85323375285137, 235.07821048680222, 259.12256986191005, 272.74061022525177, 277.89486977688716, 274.3131107577532, 295.01495757676275, 317.27052808699153, 300.16420042225275, 333.32725422104494, 343.68393141816335, 391.27759801619754, 520.807764993042, 340.22820026767306, 369.27982742435984, 592.2973348016446, 528.7048294770868, 420.5144335517208, 433.85919975190126, 411.4163056034273, 478.7412928450363, 498.7674372818395]

actual_white1_pix = [333.82632498070495, 303.4507934888327, 304.8233764908631, 343.9233655906513, 313.65928538432615, 302.4680374315356, 310.0243866176397, 301.50966799187864, 304.15765093386415, 313.72287142747484, 325.2881773277518, 301.4985593304244, 303.6812408671322, 343.7939392393406, 349.18341356950197, 352.59292911256375, 350.00714267493674, 348.97770542341414, 435.14434474280876, 388.50461735799547, 392.6233361937745, 385.9066376115487, 407.94731606147263, 423.7505800464285, 511.77025382697866]
pred_white1_pix = [320.6158199590422, 317.3537791877333, 340.31930551313815, 336.3686312015757, 313.6848114312483, 317.7481473208164, 318.63911463919254, 328.00069117506195, 321.6144471669992, 335.1076117567748, 315.52970380639107, 309.7401153701778, 333.2640251592556, 368.6811468352962, 367.26857891102577, 381.12934138102935, 392.8391403637994, 390.11984104714537, 416.5165955197247, 415.4879673997894, 402.77669529663757, 417.8928844311346, 441.3608008600618, 484.8932823451146, 485.429275904816]

actual_344_pix = [203.96452983217014, 133.61173485979035, 142.53804883772028, 147.84133442293205, 138.599221938653, 151.42764906339923, 169.2512150126973, 148.31180427814633, 150.36753236814712, 147.29646455628168, 146.68124086713192, 143.02438661763952, 139.2647287068718, 131.3969696196699, 131.3969696196699, 127.42640687119278, 132.2416885767, 140.01219330881975, 142.42640687119282, 126.2132034355964, 144.59797974644664, 140.18376618407353, 141.94112549695421, 131.69848480983495, 148.35533905932732]
pred_344_pix = [168.92950215345138, 147.9364633593551, 156.13708498984755, 154.60260518390578, 136.45057834174162, 156.88374865756083, 158.62770707975656, 139.94495755065978, 158.18814386984027, 158.5659212389093, 157.24309399477738, 155.91950314725722, 174.07485805276514, 170.23755407168554, 170.87246796489467, 129.92843250406233, 129.40537469352722, 146.78198627305767, 146.59797974644658, 137.4595192090185, 144.28877359292886, 150.96353497526871, 151.1009250738538, 160.98255816535635, 165.42468282893586]

# convert to meters from pixels
# assume the median blacktip is around 1m
# set drone height to match that, real flight metadata is unknown. Assume constant altitude and nadir angle, which is not true
actual_blacktip = [pixels_to_meters(measurement, altitude=10) for measurement in actual_blacktip_pix]
pred_blacktip = [pixels_to_meters(measurement, altitude=10) for measurement in pred_blacktip_pix]
print(f'median blacktip length: {np.median(actual_blacktip)} m')

# assume the median juvenile whiteshark should be around 3m, following the upper size limit of a juvenile white shark from literature
# set drone height to match that, real flight metadata is unknown
actual_white1 = [pixels_to_meters(measurement, altitude=57) for measurement in actual_white1_pix]
pred_white1 = [pixels_to_meters(measurement, altitude=57) for measurement in pred_white1_pix]
print(f'median white length: {np.median(actual_white1)} m')

actual_344 = [pixels_to_meters(measurement) for measurement in actual_344_pix]
pred_344 = [pixels_to_meters(measurement) for measurement in pred_344_pix]
print(f'median nurse length: {np.median(actual_344)} m')

data = {
    "Actual - White": actual_white1,
    "Predicted - White": pred_white1,
    "Actual - Nurse": actual_344,
    "Predicted - Nurse": pred_344,
    "Actual - Blacktip": actual_blacktip,
    "Predicted - Blacktip": pred_blacktip,
}
species_map = {
    "Actual - White": "White Shark",
    "Predicted - White": "White Shark",
    "Actual - Nurse": "Nurse Shark",
    "Predicted - Nurse": "Nurse Shark",
    "Actual - Blacktip": "Blacktip Reef Shark",
    "Predicted - Blacktip": "Blacktip Reef Shark",
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
percent_error_white = percent_error(actual_white1, pred_white1)
percent_error_nurse = percent_error(actual_344, pred_344)
percent_error_blacktip = percent_error(actual_blacktip, pred_blacktip)
percent_errors = np.concatenate([percent_error_white, percent_error_nurse, percent_error_nurse])

# calculate and print mean ± SD for each array
print("\n")
print(f"Blacktip (Actual) Mean ± SD = {np.mean(actual_blacktip):.2f} ± {np.std(actual_blacktip):.2f} m")
print(f"Blacktip (Predicted) Mean ± SD = {np.mean(pred_blacktip):.2f} ± {np.std(pred_blacktip):.2f} m")
print(f"White Shark (Actual) Mean ± SD = {np.mean(actual_white1):.2f} ± {np.std(actual_white1):.2f} m")
print(f"White Shark (Predicted) Mean ± SD = {np.mean(pred_white1):.2f} ± {np.std(pred_white1):.2f} m")
print(f"Nurse Shark (Actual) Mean ± SD = {np.mean(actual_344):.2f} ± {np.std(actual_344):.2f} m")
print(f"Nurse Shark (Predicted) Mean ± SD = {np.mean(pred_344):.2f} ± {np.std(pred_344):.2f} m")

print("\n")
print(f'White Shark Mean percent error ± SD: {np.mean(percent_error_white):.2f} ± {np.std(percent_error_white):.2f} %')
print(f'Nurse Shark Mean percent error ± SD: {np.mean(percent_error_nurse):.2f} ± {np.std(percent_error_nurse):.2f} %')
print(f'Blacktip Shark Mean percent error ± SD: {np.mean(percent_error_blacktip):.2f} ± {np.std(percent_error_blacktip):.2f} %')

print("\n")
print(f'White Shark percent error min: {np.min(percent_error_white):.2f} %, max: {np.max(percent_error_white):.2f} %, n = {len(percent_error_white)}')
print(f'Nurse Shark percent error min: {np.min(percent_error_nurse):.2f} %, max: {np.max(percent_error_nurse):.2f} %, n = {len(percent_error_nurse)}')
print(f'Blacktip Shark percent error min: {np.min(percent_error_blacktip):.2f} %, max: {np.max(percent_error_blacktip):.2f} %, n = {len(percent_error_blacktip)}')

print("\n")
print(f'Overall Mean percent error ± SD: {np.mean(percent_errors):.2f} ± {np.std(percent_errors):.2f} %')
print(f'Overall percent error min: {np.min(percent_errors):.2f} %, max: {np.max(percent_errors):.2f} %, n = {percent_errors.shape[0]}')

print('\n')
print(f'White shark mean ± mean absolute error: {np.mean(actual_white1):.2f} ± {mean_absolute_error(actual_white1, pred_white1):.2f} m')
print(f'Nurse shark mean ± mean absolute error: {np.mean(actual_344):.2f} ± {mean_absolute_error(actual_344, pred_344):.2f} m')
print(f'Blacktip shark mean ± mean absolute error: {np.mean(actual_blacktip):.2f} ± {mean_absolute_error(actual_blacktip, pred_blacktip):.2f} m')

# create a figure with 4 subplots, sharing equal space between scatter plot and violin plots
fig, axs = plt.subplots(1, 4, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 0.2, 0.2, 0.2]})

# --- plot 1: predicted vs actual lengths scatter plot ---
x = np.linspace(0, 8, 100)
axs[0].plot(x, x, color="grey", linestyle="--", linewidth=1, zorder=1)
axs[0].scatter(actual_white1, pred_white1, color=colors["White Shark"], label="White Shark", zorder=2)
axs[0].scatter(actual_344, pred_344, color=colors["Nurse Shark"], label="Nurse Shark", zorder=3)
axs[0].scatter(actual_blacktip, pred_blacktip, color=colors["Blacktip Reef Shark"], label="Blacktip Reef Shark", zorder=2)

axs[0].set_xlim(0, 8)
axs[0].set_ylim(0, 8)
axs[0].set_xlabel("Manual length (m)", fontsize=16)
axs[0].set_ylabel("FLAIR length (m)", fontsize=16)

axs[0].legend(title='Species', fontsize=16, title_fontproperties={"weight": "bold", "size":16})
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=12)
axs[0].set_xticklabels(axs[0].get_xticks(), fontsize=12)
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
        axs[i + 1].set_ylabel('Length (m)', fontsize=16)
    else:
        axs[i + 1].set_ylabel(None)
    axs[i + 1].set_xticklabels(["Manual", "FLAIR"], fontsize=14, fontweight='bold')
    axs[i + 1].set_yticklabels(axs[i + 1].get_yticks(), fontsize=12)
    axs[i + 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

axs[1].set_ylim(3, 8.5)
axs[2].set_ylim(1, 2.1)
axs[3].set_ylim(0, 2.2)

plt.tight_layout()
plt.savefig("figures/combined_length.png", dpi=600, bbox_inches="tight")



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





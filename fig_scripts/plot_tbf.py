# Code for plotting tailbeat frequency over time from a series of shark masks

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from utils.photogrammetry import pixels_to_meters
from utils.stats import percent_error, mean_absolute_error
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# following values are the output from tbf_eval.py

# --- Raw Lengths ---
nurse_raw_pix = np.loadtxt("fig_scripts/tbf_results/nurse/raw/flair.txt").tolist()
nurse_raw_pix_hil = np.loadtxt("fig_scripts/tbf_results/nurse/raw/hil.txt").tolist()
white_raw_pix = np.loadtxt("fig_scripts/tbf_results/white/raw/flair.txt").tolist()
white_raw_pix_hil = np.loadtxt("fig_scripts/tbf_results/white/raw/hil.txt").tolist()
blacktip_raw_pix = np.loadtxt("fig_scripts/tbf_results/blacktip/raw/flair.txt").tolist()
blacktip_raw_pix_hil = np.loadtxt("fig_scripts/tbf_results/blacktip/raw/hil.txt").tolist()


# --- Smoothed Lengths ---
nurse_smooth_pix = np.loadtxt("fig_scripts/tbf_results/nurse/smooth/flair.txt").tolist()
nurse_smooth_pix_hil = np.loadtxt("fig_scripts/tbf_results/nurse/smooth/hil.txt").tolist()
white_smooth_pix = np.loadtxt("fig_scripts/tbf_results/white/smooth/flair.txt").tolist()
white_smooth_pix_hil = np.loadtxt("fig_scripts/tbf_results/white/smooth/hil.txt").tolist()
blacktip_smooth_pix = np.loadtxt("fig_scripts/tbf_results/blacktip/smooth/flair.txt").tolist()
blacktip_smooth_pix_hil = np.loadtxt("fig_scripts/tbf_results/blacktip/smooth/hil.txt").tolist()


# --- Crossing Pairs (tuples) ---
crossing_pairs_ns_flair = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/nurse/crossing_pairs/flair.txt", dtype=int)]
crossing_pairs_ns_hil = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/nurse/crossing_pairs/hil.txt", dtype=int)]
ground_truth_pairs_ns = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/nurse/crossing_pairs/gt.txt", dtype=int)]
crossing_pairs_w_flair = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/white/crossing_pairs/flair.txt", dtype=int)]
crossing_pairs_w_hil = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/white/crossing_pairs/hil.txt", dtype=int)]
ground_truth_pairs_w = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/white/crossing_pairs/gt.txt", dtype=int)]
crossing_pairs_b_flair = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/blacktip/crossing_pairs/flair.txt", dtype=int)]
crossing_pairs_b_hil = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/blacktip/crossing_pairs/hil.txt", dtype=int)]
ground_truth_pairs_b = [tuple(row) for row in np.loadtxt("fig_scripts/tbf_results/blacktip/crossing_pairs/gt.txt", dtype=int)]


# sliding window for ground truth tailbeats
def calculate_ground_truth_tailbeats(ground_truth_pairs, start, end, window_size=120, step_size=15, fps=30): 
    expected_tailbeats = []

    for s in range(start, end - window_size + 1, step_size):
        e = s + window_size
        count = 0 # count ground truth tailbeats in the window
        for pair in ground_truth_pairs:
            if pair[0] >= s and pair[1] <= e:
                count += 1
            elif pair[0] < s and pair[1] > s:
                count += (pair[1] - s)/(pair[1] - pair[0])
            elif pair[0] < e and pair[1] > e:
                count += (e - pair[0])/(pair[1] - pair[0])
        expected_tailbeats.append(count * fps/window_size) #convert to Hz 

    return expected_tailbeats

# get times when TBF is sampled
predicted_intercross_frames_ns = [pair[1]-pair[0] for pair in crossing_pairs_ns_flair]
predicted_intercross_frames_b = [pair[1]-pair[0] for pair in crossing_pairs_b_flair]
predicted_intercross_frames_w = [pair[1]-pair[0] for pair in crossing_pairs_w_flair]
gt_intercross_frames_ns = [pair[1]-pair[0] for pair in ground_truth_pairs_ns]
gt_intercross_frames_b = [pair[1]-pair[0] for pair in ground_truth_pairs_b]
gt_intercross_frames_w = [pair[1]-pair[0] for pair in ground_truth_pairs_w]


# for frequencies (bottom graph)
# ground truth
actual_tailbeats_ns = calculate_ground_truth_tailbeats(ground_truth_pairs_ns, start=2631, end=2895)
actual_tailbeats_b = calculate_ground_truth_tailbeats(ground_truth_pairs_b, start=9, end=522)
actual_tailbeats_w = calculate_ground_truth_tailbeats(ground_truth_pairs_w, start=33, end=751)

# FLAIR
flair_expected_tailbeats_ns = calculate_ground_truth_tailbeats(crossing_pairs_ns_flair, start=2631, end=2895)
flair_expected_tailbeats_b = calculate_ground_truth_tailbeats(crossing_pairs_b_flair, start=9, end=522)
flair_expected_tailbeats_w = calculate_ground_truth_tailbeats(crossing_pairs_w_flair, start=33, end=751)

# HiL
hil_expected_tailbeats_ns = calculate_ground_truth_tailbeats(crossing_pairs_ns_hil, start=2631, end=2895)
hil_expected_tailbeats_b = calculate_ground_truth_tailbeats(crossing_pairs_b_hil, start=9, end=522)
hil_expected_tailbeats_w = calculate_ground_truth_tailbeats(crossing_pairs_w_hil, start=33, end=751)

print(max(actual_tailbeats_b))

####### displacement (top graph) range  ###########
nurse_displacement = {"start": 2620, "end":2930, "buffer":2550}
white_displacement = {"start": 0, "end":752, "buffer": 0}
blacktip_displacement = {"start": 0, "end":555, "buffer": 0}
############ frequencies (bottom graph) range ################
nurse_freq = {"start": 2631, "end":2895}
white_freq = {"start": 33, "end":751}
blacktip_freq = {"start": 9, "end":522}

# frequencies should start after displacement
fps = 30
nurse = 2895-2631
blacktip = 522-9
white = 751-33
print("tailbeats range")
print(f'nurse shark freq length {nurse/fps} seconds')
print(f'blacktip shark freq length {blacktip/fps} seconds')
print(f'white shark freq length {white/fps} seconds')

print("\n")
print("tailbeats data")
print(f'nurse shark freq length {len(actual_tailbeats_ns)} seconds')
print(f'blacktip shark freq length {len(actual_tailbeats_b)} seconds')
print(f'white shark freq length {len(actual_tailbeats_w)} seconds')

# convert to meters
nurse_raw = [pixels_to_meters(dist) for dist in nurse_raw_pix]
white_raw = [pixels_to_meters(dist, altitude=57) for dist in white_raw_pix]
blacktip_raw = [pixels_to_meters(dist, altitude=10) for dist in blacktip_raw_pix]

nurse_smooth = [pixels_to_meters(dist) for dist in nurse_smooth_pix]
white_smooth = [pixels_to_meters(dist, altitude=57) for dist in white_smooth_pix]
blacktip_smooth = [pixels_to_meters(dist, altitude=10) for dist in blacktip_smooth_pix]

nurse_smooth_hil = [pixels_to_meters(dist) for dist in nurse_smooth_pix_hil]
white_smooth_hil = [pixels_to_meters(dist, altitude=57) for dist in white_smooth_pix_hil]
blacktip_smooth_hil = [pixels_to_meters(dist, altitude=10) for dist in blacktip_smooth_pix_hil]

print("\n")
print("displacements range")
print(f'nurse shark displacement length {(nurse_displacement['end'] - nurse_displacement['start'])/fps} seconds')
print(f'blacktip shark displacement length {(blacktip_displacement['end'] - blacktip_displacement['start'])/fps} seconds')
print(f'white shark displacement length {(white_displacement['end'] - white_displacement['start'])/fps} seconds')

print("\n")
print("displacements data")
print(f'nurse shark displacement length {len(nurse_smooth)/fps} seconds')
print(f'blacktip shark displacement length {len(blacktip_smooth)/fps} seconds')
print(f'white shark displacement length {len(white_smooth)/fps} seconds')

# define time axis
time_nurse = [x / fps for x in range(len(nurse_smooth))]
time_white = [x / fps for x in range(len(white_smooth))]
time_blacktip = [x / fps for x in range(len(blacktip_smooth))]

time_nurse_hil = [(x + 5) / fps  for x in range(len(nurse_smooth_hil))]
time_white_hil = [x / fps for x in range(len(white_smooth_hil))]
time_blacktip_hil = [x / fps for x in range(len(blacktip_smooth_hil))]

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

# create the figure with subplots
fig, axs = plt.subplots(4, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1, 1, 2]}, sharex=True)

# top 3 subplots (tail displacement)
# white Shark
axs[0].plot(np.linspace(-0.5, 41, 100), np.zeros(100), linestyle='--', color='grey')
axs[0].plot(time_white, white_raw, color='grey', linewidth=0.5)
axs[0].plot(time_white, white_smooth, color=colors["White Shark"], linewidth=2)
axs[0].plot(time_white_hil, white_smooth_hil, color=colors["White Shark"], linewidth=2, linestyle='--')
axs[0].grid(False)
axs[0].set_ylim(-1.4, 1.4)
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=14)

# nurse shark
axs[1].plot(np.linspace(-0.5, 41, 100), np.zeros(100), linestyle='--', color='grey')
axs[1].plot(time_nurse, nurse_raw, color='grey', linewidth=0.5)
axs[1].plot(time_nurse, nurse_smooth, color=colors["Nurse Shark"], linewidth=2)
axs[1].plot(time_nurse_hil, nurse_smooth_hil, color=colors["Nurse Shark"], linewidth=2, linestyle='--')
axs[1].grid(False)
axs[1].set_ylim(-0.3, 0.3)
axs[1].set_ylabel("Tail displacement from body centerline (m)", fontsize=18)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=14)
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# blacktip reef shark
axs[2].plot(np.linspace(-0.5, 41, 100), np.zeros(100), linestyle='--', color='grey')
axs[2].plot(time_blacktip, blacktip_raw, color='grey', linewidth=0.5)
axs[2].plot(time_blacktip, blacktip_smooth, color=colors["Blacktip Reef Shark"], linewidth=2)
axs[2].plot(time_blacktip_hil, blacktip_smooth_hil, color=colors["Blacktip Reef Shark"], linewidth=2, linestyle='--')
axs[2].grid(False)
axs[2].set_ylim(-0.4, 0.4)
axs[2].set_yticklabels(axs[2].get_yticks(), fontsize=14)
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# bottom subplot (tailbeat frequency)
axs[3].plot([(x + (white_freq['start'] - white_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_w))], actual_tailbeats_w, color='k', label='Ground Truth', marker="o")
axs[3].plot([(x + (white_freq['start'] - white_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_w))], flair_expected_tailbeats_w, color=colors["White Shark"], marker="s", linestyle="-", label='White Shark - FLAIR')
axs[3].plot([(x + (white_freq['start'] - white_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_w))], hil_expected_tailbeats_w, color=colors["White Shark"], marker="^", linestyle="--", label='White Shark - HiL')

axs[3].plot([(x + (nurse_freq['start'] - nurse_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_ns))], actual_tailbeats_ns, color='k', marker="o")
axs[3].plot([(x + (nurse_freq['start'] - nurse_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_ns))], flair_expected_tailbeats_ns, color=colors["Nurse Shark"], marker="s", linestyle="-", label='Nurse Shark - FLAIR')
axs[3].plot([(x + (nurse_freq['start'] - nurse_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_ns))], hil_expected_tailbeats_ns, color=colors["Nurse Shark"], marker="^", linestyle="--", label='Nurse Shark - HiL')

axs[3].plot([(x + (blacktip_freq['start'] - blacktip_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_b))], actual_tailbeats_b, color='k', marker="o")
axs[3].plot([(x + (blacktip_freq['start'] - blacktip_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_b))], flair_expected_tailbeats_b, color=colors["Blacktip Reef Shark"], marker="s", linestyle="-", label='Blacktip Reef Shark - FLAIR')
axs[3].plot([(x + (blacktip_freq['start'] - blacktip_displacement['start'])/fps)/2 + 2.5 for x in range(len(actual_tailbeats_b))], hil_expected_tailbeats_b, color=colors["Blacktip Reef Shark"], marker="^", linestyle="--", label='Blacktip Reef Shark - HiL')

axs[3].grid(False)
axs[3].set_ylim(0.4, 0.9)
axs[3].set_xlim(0, 25)
axs[3].set_xlabel("Time (s)", fontsize=18)
axs[3].set_ylabel("Tailbeat frequency (Hz)", fontsize=18)

axs[3].legend(fontsize=16, loc="upper right")
axs[3].set_yticklabels(axs[3].get_yticks(), fontsize=14)
axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[3].set_xticklabels(axs[3].get_xticks(), fontsize=14)
axs[3].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.tight_layout()
plt.savefig("fig_scripts/figures/combined_tbf_plot.png", dpi=600, bbox_inches='tight')
plt.show()


# calculate and print mean ± SD for each array
print("\n")
print(f"Blacktip (Actual) Mean ± SD = {np.mean(actual_tailbeats_b):.2f} ± {np.std(actual_tailbeats_b):.2f} Hz")
print(f"Blacktip (Predicted) Mean ± SD = {np.mean(flair_expected_tailbeats_b):.2f} ± {np.std(flair_expected_tailbeats_b):.2f} Hz")

print(f"White Shark (Actual) Mean ± SD = {np.mean(actual_tailbeats_w):.2f} ± {np.std(actual_tailbeats_w):.2f} Hz")
print(f"White Shark (Predicted) Mean ± SD = {np.mean(flair_expected_tailbeats_w):.2f} ± {np.std(flair_expected_tailbeats_w):.2f} Hz")

print(f"Nurse Shark (Actual) Mean ± SD = {np.mean(actual_tailbeats_ns):.2f} ± {np.std(actual_tailbeats_ns):.2f} Hz")
print(f"Nurse Shark (Predicted) Mean ± SD = {np.mean(flair_expected_tailbeats_w):.2f} ± {np.std(flair_expected_tailbeats_w):.2f} Hz")

percent_error_white = percent_error(actual_tailbeats_w, flair_expected_tailbeats_w)
percent_error_nurse = percent_error(actual_tailbeats_ns, flair_expected_tailbeats_ns)
percent_error_blacktip = percent_error(actual_tailbeats_b, flair_expected_tailbeats_b)
percent_errors = np.concatenate([percent_error_white, percent_error_nurse, percent_error_nurse])

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
print(f'White shark mean ± mean absolute error: {np.mean(actual_tailbeats_w):.2f} ± {mean_absolute_error(actual_tailbeats_w, flair_expected_tailbeats_w):.2f} Hz')
print(f'Nurse shark mean ± mean absolute error: {np.mean(actual_tailbeats_ns):.2f} ± {mean_absolute_error(actual_tailbeats_ns, flair_expected_tailbeats_ns):.2f} Hz')
print(f'Blacktip shark mean ± mean absolute error: {np.mean(actual_tailbeats_b):.2f} ± {mean_absolute_error(actual_tailbeats_b, flair_expected_tailbeats_b):.2f} Hz')


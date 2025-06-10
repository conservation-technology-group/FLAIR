# Code for calculating tailbeat frequency from a series of shark masks

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
import cv2
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from tqdm import tqdm

from tbf_utils import (
    show_mask,
    show_points,
    show_box,
    annotate_shark_parts,
    smooth_data,
    find_zero_crossings,
    refine_zero_crossings,
    calculate_tailbeat_periods
)

####### Parameters ###########
folder_path = 'predicted_masks/video_segments_white1'
video_dir = 'trimmed_videos/images/white1'
# Image frame numbers for white shark:
start = 0
end = 752
buffer = 0
frame_rate = 30  # Frames per second
prom = 0.2 # Prominence threshold: peaks have to be greater than prom * (max peak - min trough). So larger prominence means a less noisy signal but potentially missing smaller tailbeats
correct_indices = [1]  # The correct shark mask indices (from mask pruning)
output_png = "tailbeat_detection.png"
crossings_output_file = "refined_crossings.txt"
periods_frames_output_file = "tailbeat_periods_frames.txt"
periods_seconds_output_file = "tailbeat_seconds_frames.txt"

'''
########### FOR GENERALZIABILITY VIDS #################
video_dir = "generalization/white"

# Define the directory containing the images
image_dir = video_dir

# List all files in the directory
files = os.listdir(image_dir)

# Loop through the files and rename them
for file_name in files:
    if file_name.startswith("frame_") and file_name.endswith(".jpg"):
        # Extract the new name by removing "frame_"
        new_name = file_name.replace("frame_", "")

        # Form the full paths
        old_path = os.path.join(image_dir, file_name)
        new_path = os.path.join(image_dir, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

##################################
'''


frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

files = os.listdir(folder_path) 
files = sorted(files, key=lambda x: int(x.split('_')[-1].replace('.npy', '')))

filtered_dict_list = {}
for i in tqdm(range(len(files)), desc="Filtering segments"):
    file_name = files[i]
    frame_number = int(file_name.split("_")[1].split(".")[0])
    file_path = os.path.join(folder_path, file_name)

    frame_segments_old = np.load(file_path, allow_pickle=True).item()
    filtered_dict = {key: value for key, value in frame_segments_old.items() if key in correct_indices}
    filtered_dict_list[frame_number] = filtered_dict

    del frame_segments_old
    del filtered_dict

# Get distances
dists = []
for out_frame_idx in tqdm(range(start, end), desc="Processing frames"):
    # load the saved segmentation results for the current frame from disk
    frame_segments = filtered_dict_list[out_frame_idx-buffer]

    for out_obj_id, out_mask in frame_segments.items():
        dists.append(annotate_shark_parts(out_mask, plt.gca()))

    del frame_segments

dists = np.array(dists, dtype=np.float64)
# Clean up dists so it doesn't have NaNs
for i in range(len(dists)):
    if np.isnan(dists[i]):
        # Try left
        for offset in range(1, len(dists)):
            left = i - offset
            right = i + offset

            if left >= 0 and not np.isnan(dists[left]):
                dists[i] = dists[left]
                break
            if right < len(dists) and not np.isnan(dists[right]):
                dists[i] = dists[right]
                break


# calculate periods
tailbeat_periods_frames, tailbeat_periods_seconds, smoothed_dists, refined_crossings, all_extrema = calculate_tailbeat_periods(dists, frame_rate, prominence_factor=prom)

### Plot original and smoothed data, extrema, and refined zero crossings
plt.figure(figsize=(15, 6))
plt.plot(dists, label="Original Data", alpha=0.6)
plt.plot(smoothed_dists, label="Smoothed Data", linewidth=2)

# mark zero-crossings and extrema
plt.scatter(refined_crossings, smoothed_dists[refined_crossings], color='red', label="Zero Crossings")
plt.scatter(all_extrema, smoothed_dists[all_extrema], color='green', label="Peaks and Troughs")

plt.xlabel("Frame Index")
plt.ylabel("Signed Distance")
plt.legend()
plt.title("Tailbeat Detection with Refined Zero-Crossings")
plt.savefig(output_png, dpi=300, bbox_inches='tight')  # save the plot
plt.close()  # close the figure

refined_crossings = [i + start for i in refined_crossings]

np.savetxt(crossings_output_file, refined_crossings, fmt="%.4f")
np.savetxt(periods_frames_output_file, tailbeat_periods_frames, fmt="%.4f")
np.savetxt(periods_seconds_output_file, tailbeat_periods_seconds, fmt="%.4f")


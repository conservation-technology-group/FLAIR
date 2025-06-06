# run_length_eval.py

import os
import numpy as np
import cv2
import gc
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

from shark_length_utils import (
    get_total_mask_length,
    calculate_centerline_length,
    show_enhanced_skeleton_with_path,
    show_mask,
)

input_directory = 'trimmed_videos/sampled_frames/masks/white1_length'
video_dir = 'trimmed_videos/images/white1'
folder_path = 'video_segments_HiL_white1'
gt = "white1_length_gt.txt"
pred = "white1_length_pred.txt"
start = 0
end = 753
buffer = 0
correct_indices = [1]  # The correct shark mask indices (from mask pruning)


frame_names = sorted(
    [p for p in os.listdir(video_dir) if p.lower().endswith(".jpg")],
    key=lambda p: int(os.path.splitext(p)[0])
)

# Load predicted masks
files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].replace('.npy', '')))
filtered_dict_list = {}
for file_name in tqdm(files, desc="Loading mask dicts"):
    frame_number = int(file_name.split("_")[1].split(".")[0])
    file_path = os.path.join(folder_path, file_name)
    frame_segments_old = np.load(file_path, allow_pickle=True).item()
    filtered_dict = {key: value for key, value in frame_segments_old.items() if key in correct_indices}
    filtered_dict_list[frame_number] = filtered_dict
    del frame_segments_old, filtered_dict

# Load ground truth mask PNGs
img_names = sorted([p for p in os.listdir(input_directory) if p.endswith(".png")], key=lambda p: int(p.split('.')[0]))
file_nums = [int(name.split('.')[0]) for name in img_names]
results = [calculate_centerline_length(os.path.join(input_directory, name), cv2, np, skeletonize, euclidean) for name in img_names]

# Compare to FLAIR predictions

calc_lengths = []

for out_frame_idx in tqdm(range(start, end), desc="Calculating predicted lengths"):
    if out_frame_idx not in file_nums:
        continue

    frame_segments = filtered_dict_list[out_frame_idx-buffer]
    
    for out_obj_id, out_mask in frame_segments.items():
        length = get_total_mask_length(out_mask)[0]
        calc_lengths.append(length)

        # --- Optional Visualization Code Below ---
        """
        img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx - buffer]))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)

        show_mask(out_mask, ax, obj_id=out_obj_id)
        # Uncomment this to overlay enhanced skeleton path:
        show_enhanced_skeleton_with_path(out_mask, ax)

        plt.axis('off')
        plt.tight_layout()
        plt.show()
        """
    
    del frame_segments

gc.collect()

print("Ground truth:", results)
print("Predicted:", calc_lengths)

# Save results to disk
np.savetxt(gt, results, fmt="%.4f")
np.savetxt(pred, calc_lengths, fmt="%.4f")

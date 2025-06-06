### Calculate Dice coefficient between predicted masks and ground truths                                                                                                                                                                                      
### Main testing code for different methods and shark datasets                                                                                                                                                                                                

import numpy as np
from imageio import imread
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_bounding_box(mask):

    # Find the indices where mask is True                                                                                                                                                                                                                     
    y_indices, x_indices = np.where(mask)

    # Calculate bounding box coordinates                                                                                                                                                                                                                      
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)

    return (x_min, y_min, x_max, y_max)

def dice_coefficient(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate Dice coefficient                                                                                                                                                                                                                              
    intersection = np.sum(y_true * y_pred)
    dice_score = (2 * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice_score

gt_masks = "trimmed_videos/sampled_frames/masks/video1_video2_sample_masks"
predicted_masks = "FLAIR/video_segments_video2_small_3"
buffer = 2400

all_dice_scores = []
all_masks = os.listdir(gt_masks)

for m in tqdm(gt_masks, desc="Processing masks"):

    num = int(m[7:-4]) - buffer

    s = m[0:6] + "_frame_" + str(num) + ".npy"
    #s = "frame_" + str(num) + ".npy"
    dcs = []

    if not os.path.exists(os.path.join(predicted_masks, s)):
         print("NO PREDICTED MASKS", m)
         dcs.append(0)
    else:
        frame = np.load(os.path.join(predicted_masks, s), allow_pickle=True).item()

        ground_truth_array = imread(os.path.join(gt_masks, m))
        
        # Create an empty mask with same shape as input image
        height, width, _ = ground_truth_array.shape
        mask = np.zeros((height, width), dtype=int)

        mask_color1 = np.array([250, 50, 83], dtype=np.uint8)
        mask_color2 = np.array([184, 61, 245], dtype=np.uint8)

        # Set values in mask based on the colors
        mask[np.all(ground_truth_array == mask_color1, axis=-1)] = 1  # first mask
        mask[np.all(ground_truth_array == mask_color2, axis=-1)] = 2  # second mask

        uni = set()
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                uni.add(mask[i][j])
        
        for i in uni:
            if i == 0:
                continue
            temp_mask = (mask == i).astype(int)
            temp_dc = 0
            for f in frame.keys():
                            temp_dc = max(temp_dc, dice_coefficient(temp_mask, np.squeeze(frame[f])))
            dcs.append(temp_dc)
            del temp_mask
        
        print(m, sum(dcs) / len(dcs))
        del frame, ground_truth_array

    all_dice_scores.append(sum(dcs) / len(dcs))


print(all_dice_scores)

print("Mean dice score:", sum(all_dice_scores) / len(all_dice_scores))


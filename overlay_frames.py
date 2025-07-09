### This file is for overlaying FLAIR masks back on the original images for visualization.

import gc
import os
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import cv2

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # red color
        color = np.array([1, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def polygons_to_mask(polygons, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

# Directory containing .npy files
npy_dir = 'video_segments_video2_small_5160-5760'

npy_files = [f for f in os.listdir(npy_dir) if f.startswith('frame_') and f.endswith('.npy')]

frame_numbers = [int(f.split('_')[1].split('.')[0]) for f in npy_files]

# Find minimum and maximum frame numbers
min_frame = min(frame_numbers)
max_frame = max(frame_numbers)



video_dir = "sam_dji_video2_small"

# All JPG or JPEG frames in directory
frame_names = [p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
frame_width, frame_height = first_frame.size

def save_frame_with_masks(frame_path, masks, output_path):
    """
    Overlay masks on the frame and save the result.
    """
    # Open frame image
    frame_image = Image.open(frame_path)

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(frame_image)

    # Overlay each mask on frame
    for obj_id, mask in masks.items():
        show_mask(mask, ax, obj_id=obj_id)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    plt.clf() 
    plt.close(fig)

    frame_image.close()
    del frame_image
    gc.collect()


# Saving overlaid frames
output_frame_dir = 'overlaid_frames_video2_small_new'
os.makedirs(output_frame_dir, exist_ok=True)

for frame_idx in range(min_frame, max_frame+1, 5):
    if frame_idx % 50 == 0:
      print(f"Processing frame {frame_idx}")

    # Load masks from disk for current frame
    mask_path = f'{npy_dir}/frame_{frame_idx}.npy'
    if not os.path.exists(mask_path):
        print(f"Mask for frame {frame_idx} not found.")
        continue

    # Load saved segmentation masks for the frame
    mask_polygons = np.load(mask_path, allow_pickle=True).item()
    masks = {k: polygons_to_mask(v, image.shape[:2]) for k, v in mask_polygons.items()}

    frame_path = os.path.join(video_dir, frame_names[frame_idx])

    # Output path for overlaid frame
    output_path = os.path.join(output_frame_dir, f'{frame_idx:05d}.jpg')

    # Save frame with overlaid masks
    save_frame_with_masks(frame_path, masks, output_path)

    del masks
    gc.collect()

### Sample images from a larger directory for biometrics and downstream tasks

import os
import random
import shutil

# Define source and target folders
source_folder = "generalization/white1"
target_folder = "white1_sample" 

os.makedirs(target_folder, exist_ok=True)

start_range = 0 
end_range = 752

# Get all image filenames in source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]

filtered_images = [
    img for img in image_files
    if img.endswith(".jpg") and start_range <= int(img.split(".")[0]) <= end_range
]

# Randomly sample 25 images
sampled_images = random.sample(filtered_images, 25)

# Copy sampled images to the target folder
for img in sampled_images:
    src_path = os.path.join(source_folder, img)
    dest_path = os.path.join(target_folder, img)
    shutil.copy(src_path, dest_path)

print(f"Sampled 25 images from {start_range} to {end_range} and saved them in {target_folder}.")

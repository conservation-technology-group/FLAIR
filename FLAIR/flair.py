#!/usr/bin/env python3
import os
import yaml
import gc
import pickle
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
#import torchvision
import argparse


from sam_setup import setup_device, load_sam_predictor, load_mask_generator
from utils import *
from clip_utils import load_clip_model
from generate_mask_grid import generate_mask_grid
from tracking_propagation import run_propagation

print("Torch is cuda available:", torch.cuda.is_available())
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Load config
# Parse config file path from command-line
parser = argparse.ArgumentParser(description="Run FLAIR with a specified config file.")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
args = parser.parse_args()

# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

device = setup_device(config["sam2"]["device"])
print("Using device:", device)

# Load SAM2 and CLIP
predictor = load_sam_predictor(config, device)
mask_generator = load_mask_generator(config, device)
clip_processor, clip_model = load_clip_model()

# === Prepare video frames ===
video_dir = config["paths"]["video_dir"]
frame_gap = config["params"]["run_every_n_frames"]
frame_names = sorted([
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
], key=lambda p: int(os.path.splitext(p)[0]))

images = [i for i in range(0, len(frame_names), frame_gap)]

# === Mask generation + bounding boxes ===
all_frames_bboxes = []
for i, img in enumerate(tqdm(images)):
    img = np.array(Image.open(os.path.join(video_dir, frame_names[img])).convert("RGB"))
    masks = mask_generator.generate(img)
    masks = keep_masks(masks, config["params"]["min_mask_length"], config["params"]["max_mask_length"])
    boxes = bboxes_from_masks(masks, img.shape, config["params"]["bbox_buffer"])
    all_frames_bboxes.append(boxes)
    torch.cuda.empty_cache()

# === CLIP filtering ===
keep_boxes = []
prompts = config["clip"]["prompts"]

for i, (img, bboxes) in enumerate(zip(images, all_frames_bboxes)):
    img = np.array(Image.open(os.path.join(video_dir, frame_names[img])).convert("RGB"))
    
    crops = [img[y1:y2+1, x1:x2+1] for x1, y1, x2, y2 in bboxes]
    if not crops: continue

    inputs = clip_processor(text=prompts, images=crops, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs["logits_per_image"].softmax(dim=1).detach().numpy()

    shark_probs = [p[config["clip"]["clip_correct"]] for p in probs]
    for j, probs in enumerate(shark_probs):
        for prob in probs:
          if prob > config["clip"]["clip_confidence"]:
              keep_boxes.append((i, j, bboxes[j]))

# === Save to .txt ===
txt_out = config["paths"]["output_txt"]
with open(txt_out, "w") as f:
    for i, j, bbox in keep_boxes:
        f.write(f"{i},{j},{','.join(map(str, bbox))}\n")

print(f"Saved {len(keep_boxes)} boxes to {txt_out}")

# === Reorganize to frame_keep_boxes ===
frame_keep_boxes = [[] for _ in range(len(images))]
for i, j, bbox in keep_boxes:
    frame_keep_boxes[i].append((i, j, bbox))

# === Run tracking and propagation ===
print("Setting inference state...")
image = np.array(Image.open(os.path.join(video_dir, frame_names[0])).convert("RGB"))
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=True)


window_length = config["params"]["window_length"]
output_dir = config["paths"]["output_dir_prefix"]

run_propagation(
    predictor=predictor,
    inference_state=inference_state,
    frame_keep_boxes=frame_keep_boxes,
    frame_names=frame_names,
    video_dir=video_dir,
    image_shape=image.shape,
    frame_gap=frame_gap,
    window_length=window_length,
    output_dir=output_dir,
    bbox_buffer=config["params"]["bbox_buffer"]
)

generate_mask_grid(
    n_per_mask=config["mask_plotting"]["n_per_mask"],
    resize_factor=config["mask_plotting"]["resize_factor"],
    mask_dir=config["paths"]["output_dir_prefix"],
    video_dir=config["paths"]["video_dir"],
    image_shape=image.shape,
    output_pdf_path=config["paths"]["output_pdf_path"]
)



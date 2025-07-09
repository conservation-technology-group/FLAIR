import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

def get_bounding_box(mask):
    mask = np.squeeze(mask)
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    return (np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices))

def generate_mask_grid(n_per_mask, resize_factor, mask_dir, video_dir, image_shape, output_pdf_path):
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    all_samples = defaultdict(list)
    for fname in tqdm(os.listdir(mask_dir), desc="Indexing masks"):
        if not fname.endswith(".npy"):
            continue
        frame_index = int(fname.split("_")[1].split(".")[0])
        fpath = os.path.join(mask_dir, fname)
        data = np.load(fpath, allow_pickle=True).item()
        for mask_num, mask_polygons in data.items():
            # Convert polygon back to numpy mask for get_bounding_box
            mask = polygons_to_mask(mask_polygons, (image_shape[0], image_shape[1]))  # You'll need to pass image_shape
            bbox = get_bounding_box(mask)
            if bbox is not None:
                all_samples[mask_num].append((frame_index, bbox))
        del data
        gc.collect()

    with PdfPages(output_pdf_path) as pdf:
        for mask_num, samples in all_samples.items():
            if len(samples) < n_per_mask:
                continue
            selected = random.sample(samples, n_per_mask)
            frame_indices = [f for f, _ in samples]

            fig, axes = plt.subplots(3, 3, figsize=(12, 9))
            axes = axes.flatten()
            for ax, (frame_index, bbox) in zip(axes, selected):
                img_path = os.path.join(video_dir, frame_names[frame_index])
                image = Image.open(img_path).convert("RGB")
                if resize_factor != 1.0:
                    image = image.resize(
                        (int(image.width * resize_factor), int(image.height * resize_factor))
                    )
                    scale = resize_factor
                else:
                    scale = 1.0
                img_np = np.array(image)
                bbox_scaled = tuple(int(c * scale) for c in bbox)

                pad = 5
                x_min = max(bbox_scaled[0] - pad, 0)
                y_min = max(bbox_scaled[1] - pad, 0)
                x_max = min(bbox_scaled[2] + pad, img_np.shape[1])
                y_max = min(bbox_scaled[3] + pad, img_np.shape[0])

                ax.imshow(img_np)
                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.set_title(f"Frame {frame_index}")
                ax.axis("off")
                del image, img_np

            earliest = min(frame_indices)
            latest = max(frame_indices)
            fig.suptitle(f"Mask #{mask_num} | Frames {earliest}–{latest}", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()

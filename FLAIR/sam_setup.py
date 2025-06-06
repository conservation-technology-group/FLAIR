import torch
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os
import urllib.request


def download_checkpoint_if_missing(checkpoint_path):
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        print(f"Downloading SAM2 checkpoint to {checkpoint_path}...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        print("Download complete.")

def setup_device(preference='auto'):
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")

def load_sam_predictor(config, device):
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    checkpoint_path = config["sam2"]["checkpoint"]
    download_checkpoint_if_missing(checkpoint_path)
    predictor = build_sam2_video_predictor(
        config["sam2"]["model_cfg"],
        config["sam2"]["checkpoint"],
        device=device,
        vos_optimized=config["sam2"].get("vos_optimized", True)
    )
    return predictor

def load_mask_generator(config, device):
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    sam = build_sam2(
        config["sam2"]["model_cfg"],
        config["sam2"]["checkpoint"],
        device=device,
        apply_postprocessing=False
    )
    return SAM2AutomaticMaskGenerator(model=sam, **config["mask_generator"])

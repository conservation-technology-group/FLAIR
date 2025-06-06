import os
import numpy as np
from tqdm import tqdm
import argparse
import gc

parser = argparse.ArgumentParser(description="Prune .npy mask files by key.")
parser.add_argument("--mask_dir", type=str, required=True, help="Input directory with mask files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save pruned masks.")
parser.add_argument("--keep_keys", type=int, nargs="+", required=True, help="List of keys to keep.")

args = parser.parse_args()
keep_keys = set(args.keep_keys)  # use a set for faster lookup

os.makedirs(args.output_dir, exist_ok=True)

for fname in tqdm(os.listdir(args.mask_dir), desc="Filtering mask files"):
    if not fname.endswith(".npy"):
        continue

    fpath = os.path.join(args.mask_dir, fname)
    data = np.load(fpath, allow_pickle=True).item()

    filtered = {k: v for k, v in data.items() if k in keep_keys}

    outpath = os.path.join(args.output_dir, fname)
    np.save(outpath, filtered)

    del data, filtered

gc.collect()

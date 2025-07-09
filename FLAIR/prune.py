import os
import numpy as np
from tqdm import tqdm
import argparse
import gc
import sys

parser = argparse.ArgumentParser(description="Prune .npy mask files by key.")

parser.add_argument("--mask_dir", type=str, required=True, help="Input directory with mask files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save pruned masks.")
parser.add_argument("--keep_keys", type=int, nargs="+", help="List of keys to KEEP.")
parser.add_argument("--remove_keys", type=int, nargs="+", help="List of keys to REMOVE.")

args = parser.parse_args()

# Validation: Only one of keep_keys or remove_keys should be provided
if (args.keep_keys is None and args.remove_keys is None) or (args.keep_keys and args.remove_keys):
    print("Error: You must specify exactly one of --keep_keys or --remove_keys.", file=sys.stderr)
    sys.exit(1)

os.makedirs(args.output_dir, exist_ok=True)

for fname in tqdm(os.listdir(args.mask_dir), desc="Filtering mask files"):
    if not fname.endswith(".npy"):
        continue

    fpath = os.path.join(args.mask_dir, fname)
    data = np.load(fpath, allow_pickle=True).item()

    if args.keep_keys is not None:
        keys_to_keep = set(args.keep_keys)
        filtered = {k: v for k, v in data.items() if k in keys_to_keep}
    else:
        keys_to_remove = set(args.remove_keys)
        filtered = {k: v for k, v in data.items() if k not in keys_to_remove}

    outpath = os.path.join(args.output_dir, fname)
    np.save(outpath, filtered)

    del data, filtered

gc.collect()

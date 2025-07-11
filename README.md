# FLAIR: Frame Level ALIgment and tRacking
<div align="center">
  <img src="./media/output_example_video.gif" width="49.5%" style="margin:0;padding:0;border:none;">
  <img src="./media/overlaid_frames_zebra.gif" width="49.5%" style="margin:0;padding:0;border:none;"><br>
  <img src="./media/output_blacktip.gif" width="49.5%" style="margin:0;padding:0;border:none;">
  <img src="./media/overlaid_frames_output2.gif" width="49.5%" style="margin:0;padding:0;border:none;">
</div>


- We introduce Frame Level ALIgment and tRacking (FLAIR), which leverages the video understanding of Segment Anything Model 2 (SAM2) and the vision-language capabilities of Contrastive Language-Image Pre-training (CLIP)
- FLAIR takes a drone video as input and outputs segmentation masks of the species of interest across the video
- Leverages a *zero-shot* approach, eliminating the need for labeled data, training a new model, or fine-tuning an existing model to generalize to other species
- Readily generalizes to other shark species without additional human effort
- Can be combined with novel heuristics to automatically extract relevant information including length and tailbeat frequency
- FLAIR also requires markedly less human effort and expertise than traditional machine learning workflows, while achieving superior accuracy



## Quick start
You can run the FLAIR pipeline directly in Google Colab for easy setup and visualization: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/conservation-technology-group/FLAIR/blob/main/FLAIR.ipynb)

For optimal performance, please see [Step-by-step usage](#step-by-step-usage). 


## Table of contents

![FLAIR Pipeline](./media/figures/FLAIR_method.png)

- [Installation](#installation)
- [Step-by-step usage](#step-by-step-usage)
  - [Step 0: Configure parameters](#step-0-configure-parameters)
  - [Step 1: FLAIR](#step-1-flair)
  - [Step 2: Mask Pruning](#step-2-mask-pruning)
- [Citing](#citing)

---

## Installation
```bash
# (1) Download FLAIR from GitHub
$ git clone https://github.com/conservation-technology-group/FLAIR.git
$ cd FLAIR

# (2) Resolve dependencies
# Necessary packages can be installed using the following environment and requirements files.
# We strongly recommend using conda to install dependencies.
$ conda env create -f conda_env.yaml

# (3) Activate conda environment
$ conda activate FLAIR

# (4) Install pip requirements
$ pip install -r requirements.txt
```

## Step by step usage

We currently support running FLAIR on CUDA-enabled GPUs (i.e. NVIDIA). Mac MPS support will be coming soon! Running on CPU is not recommended. 

### Step 0: Configure parameters
Set the following parameters in `config.yaml`

#### Paths
* `video_dir`: Input directory containing individual image frames  
* `output_txt`: Output directory for predicted bounding boxes  
* `output_dir_prefix`: Output directory for predicted masks  
* `output_pdf_path`: Output pdf for visualizing predicted masks  

---

#### SAM2 Params
* `run_every_n_frames`: Parameter for running FLAIR every __ frames – typically equal to fps (30)  
* `min_mask_length`: Minimum object size in pixels (50)  
* `max_mask_length`: Maximum object size in pixels (150)  
* `window_length`: Window length for SAM2 Video Propagation (3)  

---

#### CLIP
* `prompts`: List of CLIP prompts to use  
* `min_mask_length`: Indices of the correct CLIP prompts  


**Additional parameters can be found in config.yaml**

---

### Step 1: FLAIR

Parameters should be set in `config.yaml` prior to running

```bash
# Run FLAIR with the specified config file (we strongly recommend utilizing a GPU)
$ python3 flair.py --config config.yaml
```

Masks in each frame are saved as polygon coordinates in a .npy file.


---


### Step 2: Mask Pruning

```bash
# Perform manual mask pruning by keeping or removing specific keys to retain correct objects of interest
# Exactly one of --keep_keys or --remove_keys must be specified
$ python3 prune.py --mask_dir --output_dir --keep_keys --remove_keys

# Example: Keep only keys 1 and 2
$ python3 prune.py --mask_dir path/to/input --output_dir path/to/output --keep_keys 1 2

# Example: Remove keys 3 and 4
$ python3 prune.py --mask_dir path/to/input --output_dir path/to/output --remove_keys 3 4
```

---

#### Paths
* `mask_dir`: Input directory containing predicted masks
* `output_dir`: Output directory containing pruned masks
* `keep_keys`: List of mask keys to keep - selected after manual review of output_pdf_path
* `remove_keys`: List of mask keys to remove - selected after manual review of output_pdf_path




## Citing
If you have used FLAIR in your work, please consider citing our paper!

```bibtex
@misc{lalgudi2025zeroshotsharktrackingbiometrics,
      title={Zero-shot Shark Tracking and Biometrics from Aerial Imagery}, 
      author={Chinmay K Lalgudi and Mark E Leone and Jaden V Clark and Sergio Madrigal-Mora and Mario Espinoza},
      year={2025},
      eprint={2501.05717},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.05717}, 
}
```

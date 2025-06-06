### Script to train YOLO model

import os
import torch
from ultralytics import YOLO


# Set environment variables
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
os.environ['CLEARML_API_ACCESS_KEY'] = 'VDF4HX9J46ONKHIU5MVD'
os.environ['CLEARML_API_SECRET_KEY'] = 'J9bynMTPj4je0bdqxL7AILNq3ojHTmQFd8abbRVnsQDmmBLwwH'


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load the model.
model = YOLO('yolov8m.pt')
model.to('cuda')

# Training.
results = model.train(
   data='sharks.yaml',
   imgsz=1080,
   epochs=100,
   batch=16,
   name='yolov8n_v8_50e_m_video2_video3_video4_video5_vdeo6',
   verbose=True,
   plots=True,
   device='cuda'
)


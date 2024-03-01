from Object_PoseEstimate_v8 import PoseEstimate
from Object_send_server import DetectSendor
from modules import VideoCaptureHard as cap

import time
import torch
from torchvision.transforms import Compose

import cv2
import numpy as np
import os

from pyskl.apis import inference_recognizer, init_recognizer
import mmcv

import queue
import threading
from collections import defaultdict
import datetime

# -------------------------------------------------------------------------------
# pose estimate
PoseEstimator = PoseEstimate(device = f"cuda:{device_use_Pose}",
                            engine_file_path=f"weights/yolov8s-pose.pt")
t1_pose = time.time()
outputs = PoseEstimator.poseTrack(imgs = imgs)
t2_pose = time.time()
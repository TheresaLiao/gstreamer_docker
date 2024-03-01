from Object_PoseEstimate_trt import PoseEstimate,posePreprocessTRT,output_to_json_TRT
from modules import poseTracking_module
import time
import cv2
import numpy as np
import os

batch=12
imgTensors = np.zeros((batch, 3, 832, 832)).astype(np.float32)

count = 0 ; maxcount=100
ts = []
img = cv2.imread("sources/img.png")

while True:
    t1 = time.time()
    for i in range(batch):
        imgTensors[i]=posePreprocessTRT(img)
    t2 = time.time()
    if count>=maxcount:
        break
    if count >=5:
        ts.append(t2-t1)
    count+=1
    time.sleep(0.02)
print("Normal mode :",np.mean(ts))


time.sleep(3)

#=========================================================#
from joblib import Parallel, delayed
backend = "threading"

def preprocessPose(imgIndex,imgRaw):
    global imgTensors
    imgTensors[i]=posePreprocessTRT(imgRaw)
        




imgTensors = np.zeros((batch, 3, 832, 832)).astype(np.float32)
img = cv2.imread("sources/img.png")
imgs = [img]*batch
count = 0
while True:
    t1 = time.time()

    Parallel(n_jobs=batch,backend = backend)(delayed(preprocessPose)(imgIndex, img) 
                                                                    for imgIndex, img in enumerate(imgs))

    t2 = time.time()
    if count >=5:
        ts.append(t2-t1)
    count+=1
    time.sleep(0.02)
    if count>=maxcount:
        break  
print("Thread mode :",np.mean(ts))

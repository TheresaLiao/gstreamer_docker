device_use_Pose = 0

from yolov7.Object_PoseEstimate import PoseEstimate,imgs2Tensor,output_to_json
from modules import poseTracking_module
from vidgear.gears import CamGear

import time
import torch
import cv2
import numpy as np
PoseEstimator = PoseEstimate(device = f"cuda:{device_use_Pose}")

def posePreprocess(lineIndex,img):
    global imgTensors
    imgTensor = imgs2Tensor(img)
    imgTensors[lineIndex] = imgTensor

objectTrackDicts = []
for camIndex in range(5):
    objectTrackDicts.append(poseTracking_module.Sort(height_max = 2160,
                                                    width_max = 3840,
                                                    categories = [],
                                                    record_absence = False,
                                                    output_absence = False,
                                                    min_hits=2,max_miss=5,
                                                    max_history = 100,
                                                    camIndex = camIndex
                                                    ))

count = 0
stream = CamGear("./sources/ntu_sample.avi").start()
while True:
    frame = stream.read()
    if frame is not None:
        img = frame
    t1_pose = time.time()
    imgTensors = torch.empty((5, 3, 576, 960), dtype=torch.float32)
    img = cv2.imread("sources/img.png")
    for i in range(5):
        posePreprocess(i,img)
    imgTensors = imgTensors.half().to(device_use_Pose)
    print(imgTensors.shape)
    t2_pose = time.time()
    outputs = PoseEstimator.posePred(imgTensors = imgTensors)
    t3_pose = time.time()

    for i,output in enumerate(outputs):
        output = PoseEstimator.nmsFromTensor(output.unsqueeze(0))
        output = output_to_json(output,ratioH = 1, ratioW = 1)
        poseTrackResult = objectTrackDicts[i].tracking(output)

        exists_id = []
        for trackIndex, track in enumerate(poseTrackResult):
            kpt = [] ; kptScore = []
            for trajectIndex in range(len(track["trajectories_opt"])):
                # print(track["trajectories_opt"][trajectIndex].keys())
                try:
                    kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
                    kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
                except:
                    kpt.append(np.zeros((17,2)))
                    kptScore.append(np.zeros(17))
            poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
            poseTrackResult[trackIndex]["keypointScore_seq"] = np.array(kptScore)
    t4_pose = time.time()


    print(f"input : {imgTensors.shape} output : {outputs.shape}")
    print(f'''
                    POSE total  cost : {round(t4_pose-t1_pose,4)} sec
                    preprocess  cost : {round(t2_pose-t1_pose,4)} sec
                    inference   cost : {round(t3_pose-t2_pose,4)} sec
                    postprocess cost : {round(t4_pose-t3_pose,4)} sec
            ''')
            
    count += 1
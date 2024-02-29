device_use_Pose = 0
batch = 12
from Object_PoseEstimate_v8 import PoseEstimate,output_to_json_V8
from modules import poseTracking_module

import time
import torch
import cv2
import numpy as np
PoseEstimator = PoseEstimate(device = f"cuda:{device_use_Pose}")


objectTrackDicts = []
for camIndex in range(batch):
    objectTrackDicts.append(poseTracking_module.Sort(height_max = 576,
                                                    width_max = 960,
                                                    categories = [],
                                                    record_absence = False,
                                                    output_absence = False,
                                                    min_hits=2,max_miss=5,
                                                    max_history = 100,
                                                    camIndex = camIndex
                                                    ))

count = 0

while True:

    t1_pose = time.time()
    img = cv2.imread("sources/img.png")
    t2_pose = time.time()

    outputs = PoseEstimator.posePred(imgs = [img]*batch)
    t3_pose = time.time()

    for i,output in enumerate(outputs):
        poseTrackResult = objectTrackDicts[i].tracking(output_to_json_V8(output))

        exists_id = []
        for trackIndex, track in enumerate(poseTrackResult):
            kpt = [] ; kptScore = []
            for trajectIndex in range(len(track["trajectories_opt"])):
                try:
                    kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
                except:
                    kpt.append(np.zeros((17,3)))
            poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
            # poseTrackResult[trackIndex]["keypointScore_seq"] = np.array(kptScore)
    t4_pose = time.time()


    print(f'''
                    POSE total  cost : {round(t4_pose-t1_pose,4)} sec
                    preprocess  cost : {round(t2_pose-t1_pose,4)} sec
                    inference   cost : {round(t3_pose-t2_pose,4)} sec
                    postprocess cost : {round(t4_pose-t3_pose,4)} sec
            ''')
            
    count += 1
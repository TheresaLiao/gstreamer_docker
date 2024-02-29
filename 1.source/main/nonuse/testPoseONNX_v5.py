device_use_Pose = 0

from Object_PoseEstimate_onnx import PoseEstimateONNX,output_to_json_onnx,Resize
from modules import poseTracking_module

import time
import torch
import cv2
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(device_use_Pose)
providers = [
    ('CUDAExecutionProvider', {'device_id': device_use_Pose,}),
     'CPUExecutionProvider',
]


batch = 12
modelPath = f"./weights/yolov5_onnx_models/Yolov5s6_pose_640_ti_litebatch{batch}.onnx"
PoseEstimator = PoseEstimateONNX(modelPath = modelPath,
                                 batchSize = batch,
                                 providers = providers
)


def posePreprocessONNX(img):
    # img = cv2.resize(img0, (640,640), interpolation=cv2.INTER_LINEAR)
    img = Resize(img,(640,640),False)
    img = (img - 127.5)/127.5 
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img

objectTrackDicts = []
for camIndex in range(batch):
    objectTrackDicts.append(poseTracking_module.Sort(height_max = 640,
                                                    width_max = 640,
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
    imgTensors = np.zeros((batch, 3, 640, 640)).astype(np.float32)
    img = cv2.imread("sources/img.png")
    for i in range(batch):
        imgTensors[i]=posePreprocessONNX(img)

    t2_pose = time.time()
    outputs = PoseEstimator.posePred(imgs = imgTensors)
    t3_pose = time.time()
    # print(imgTensors[0].shape,t3_pose-t1_pose,t3_pose-t2_pose,t2_pose-t1_pose)
    for i,output in enumerate(outputs):
        output = output_to_json_onnx(output,ratioH = 1, ratioW = 1,detConf=0.3)
        poseTrackResult = objectTrackDicts[i].tracking(output)

        exists_id = []
        for trackIndex, track in enumerate(poseTrackResult):
            kpt = [] ; kptScore = []
            for trajectIndex in range(len(track["trajectories_opt"])):
                # print(track["trajectories_opt"][trajectIndex].keys())
                try:
                    kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
                    # kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
                except:
                    kpt.append(np.zeros((17,3)))
                    kptScore.append(np.zeros(17))
            poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
            # print(np.array(kpt).shape)
    t4_pose = time.time()


    print(f"input : {imgTensors.shape} output : {len(outputs)}")
    print(f'''
    POSE total  cost : {round(t4_pose-t1_pose,4)} sec
    preprocess  cost : {round(t2_pose-t1_pose,4)} sec
    inference   cost : {round(t3_pose-t2_pose,4)} sec
    tracking    cost : {round(t4_pose-t3_pose,4)} sec
            ''')
            
    count += 1
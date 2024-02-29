device_use_Pose = 0
batch = 12
from yolov7.Object_PoseEstimate import PoseEstimate,imgs2Tensor,output_to_json
from modules import poseTracking_module

import time
import torch
import cv2
import numpy as np
PoseEstimator = PoseEstimate(device = f"cuda:{device_use_Pose}")


def Resize(img_cpu,sizeNew,cuda_cv = False):
    if cuda_cv:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(img_cpu)
            gpu_frame = cv2.cuda.resize(gpu_frame,sizeNew)
            img_cpu = gpu_frame.download()
        except:
            img_cpu = cv2.resize(img_cpu,sizeNew)

    else:
        img_cpu = cv2.resize(img_cpu,sizeNew)
    return img_cpu

def posePreprocess(img):
    return imgs2Tensor(img)

def posePreprocessV2(img):
    img = Resize(img,(960,576),False)/255
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)#.to(f"cuda:{device_use_Pose}").half()

def posePreprocessV3(img):
    return imgs2Tensor(torch.from_numpy(img.transpose(2, 0, 1)/255))#.to(f"cuda:{device_use_Pose}")


objectTrackDicts = []
for camIndex in range(batch):
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
imgTensors = torch.empty((batch, 3, 576, 960), dtype=torch.float32)#.to(device_use_Pose).half()

while True:

    t1_pose = time.time()
    img = cv2.imread("sources/img.png")
    for i in range(batch):
        # imgTensors[i] = posePreprocess(img)
        imgTensors[i] = posePreprocessV2(img)
        # posePreprocessV3(i,img)

    # imgTensors = imgTensors
    # print(imgTensors.shape)
    t2_pose = time.time()
    outputs = PoseEstimator.posePred(imgTensors = imgTensors.to(device_use_Pose).half())
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
                    # kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
                except:
                    kpt.append(np.zeros((17,3)))
                    # kptScore.append(np.zeros(17))
            poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
            # poseTrackResult[trackIndex]["keypointScore_seq"] = np.array(kptScore)
    t4_pose = time.time()


    print(f"input : {imgTensors.shape} output : {outputs.shape}")
    print(f'''
                    POSE total  cost : {round(t4_pose-t1_pose,4)} sec
                    preprocess  cost : {round(t2_pose-t1_pose,4)} sec
                    inference   cost : {round(t3_pose-t2_pose,4)} sec
                    postprocess cost : {round(t4_pose-t3_pose,4)} sec
            ''')
            
    count += 1
device_use_Pose = 0
device_use_motion = 0

video_list = [
                "./sources/ntu_sample.avi",
                "./sources/ntu_sample.avi",
                "./sources/ntu_sample.avi",
                "./sources/ntu_sample.avi",
                "./sources/ntu_sample.avi",
            ]

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

def GearAnnouce_from_sourcesList(sourcesList):
    streams = [None]*len(sourcesList)
    latency = 0
    for sourceIndex,source in enumerate(sourcesList):
        if "rtsp" in source:
            g_source = f"rtspsrc location={source} latency={latency} ! rtph264depay ! h264parse ! avdec_h264 ! cudaupload ! cudaconvert ! cudadownload ! appsink"
            streams[sourceIndex] = CamGear(source=g_source, logging=True).start()
        else:
            streams[sourceIndex] = CamGear(source).start()
    return streams

def GearRead_multiSource(streamsItems,framesBuffer):
    # t1_gearGetFrame = time.time()
    for streamIndex,stream in enumerate(streamsItems):
        frame = stream.read()
        if frame is not None:
            framesBuffer[streamIndex] = frame
        else:
            print(f"streamIndex:{streamIndex} get new frame Unsuccessful")
            print("reload..........")
            streamsItems[streamIndex].stop()
            if "rtsp" in video_list[streamIndex]:
                g_source = f"rtspsrc location={video_list[streamIndex]} latency={latency} ! rtph264depay ! h264parse ! avdec_h264 ! cudaupload ! cudaconvert ! cudadownload ! appsink"
                streamsItems[streamIndex] = CamGear(source=g_source, logging=True).start()
            else:
                streamsItems[streamIndex] = CamGear(source=video_list[streamIndex]).start()
            break
    # t2_gearGetFrame = time.time()
    # print(f"Gear Get Frame from multi source cost : {t2_gearGetFrame-t1_gearGetFrame}")
    return framesBuffer

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

# streams = GearAnnouce_from_sourcesList(video_list)
imgs = [np.zeros((2160,3840,3)).astype('uint8')] * len(video_list)
imgTensors = torch.empty((len(video_list), 3, 576, 960), dtype=torch.float32)

count = 0
while True:
    '''11111111111111111111111111111111111111'''
    for i in range(5):
        imgs[i] = cv2.imread("sources/img.png")
        
    '''222222222222222222222222222222222222222'''
    # imgs = GearRead_multiSource(streams,imgs)

    t1_pose = time.time()

    for lineIndex,img in enumerate(imgs):
        posePreprocess(lineIndex,img)
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
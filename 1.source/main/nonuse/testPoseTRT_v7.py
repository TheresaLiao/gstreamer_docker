device_use_Pose = 0


from Object_PoseEstimate_trt import PoseEstimate,posePreprocessTRT
from modules import poseTracking_module
import time
import cv2
import numpy as np
import os
batch = 5
acc = "int8"#"fp16"

# PoseEstimator = PoseEstimate(engine_file_path=f"weights/yolov7_trt_models/yolov7-w6-pose-FP16-dynamic.engine",
#                              device=0,batch = batch,structure=acc)

PoseEstimator = PoseEstimate(engine_file_path=f"weights/yolov7_trt_models/yolov7-w6-pose-dynamic_INT8.engine",
                             device=0,batch = batch,structure=acc)




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



objectTrackDicts = []
for camIndex in range(batch):
    objectTrackDicts.append(poseTracking_module.Sort(height_max = 832,
                                                    width_max = 832,
                                                    categories = [],
                                                    record_absence = False,
                                                    output_absence = False,
                                                    min_hits=2,max_miss=5,
                                                    max_history = 100,
                                                    camIndex = camIndex
                                                    ))

count = 0
imgTensors = np.zeros((batch, 3, 832, 832)).astype(np.float32)
while True:
    t1_pose = time.time()
    
    img = cv2.imread("sources/img.png")
    for i in range(batch):
        imgTensors[i]=posePreprocessTRT(img)

    t2_pose = time.time()
    outputs = PoseEstimator.posePred(imgs = imgTensors) # nms_with_model
    t3_pose = time.time()
    # print(imgTensors[0].shape,t3_pose-t1_pose,t3_pose-t2_pose,t2_pose-t1_pose)
    for i,output in enumerate(outputs):
        poseTrackResult = objectTrackDicts[i].tracking(output)

        exists_id = []
        for trackIndex, track in enumerate(poseTrackResult):
            kpt = [] ; #kptScore = []
            for trajectIndex in range(len(track["trajectories_opt"])):
                # print(track["trajectories_opt"][trajectIndex].keys())
                try:
                    kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
                    # kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
                except:
                    kpt.append(np.zeros((17,3)))
                    # kptScore.append(np.zeros(17))
            poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
            # print(np.array(kpt).shape)
    t4_pose = time.time()


    print(f"input : {imgTensors.shape} ")
    print(f'''
    POSE total  cost : {round(t4_pose-t1_pose,4)} sec
    preprocess  cost : {round(t2_pose-t1_pose,4)} sec
    inference   cost : {round(t3_pose-t2_pose,4)} sec
    postprocess cost : {round(t4_pose-t3_pose,4)} sec
            ''')
            
    count += 1

##################################################################################################
# count = 0
# imgTensors = np.zeros((batch, 3, 832, 832)).astype(np.float32)
# while True:
#     t1_pose = time.time()
    
#     img = cv2.imread("sources/img.png")
#     for i in range(batch):
#         imgTensors[i]=posePreprocessTRT(img)

#     t2_pose = time.time()
#     det_boxes,det_pose,det_scores = PoseEstimator.posePred(imgs = imgTensors) # nms_with_model
    
#     t3_pose = time.time()
#     # print(imgTensors[0].shape,t3_pose-t1_pose,t3_pose-t2_pose,t2_pose-t1_pose)
#     for bboxes, kpts, scores in zip(det_boxes, det_pose, det_scores):

#         output = output_to_json_TRT(bboxes, kpts, scores,detConf=0.3)
#         poseTrackResult = objectTrackDicts[i].tracking(output)

#         exists_id = []
#         for trackIndex, track in enumerate(poseTrackResult):
#             kpt = [] ; kptScore = []
#             for trajectIndex in range(len(track["trajectories_opt"])):
#                 # print(track["trajectories_opt"][trajectIndex].keys())
#                 try:
#                     kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
#                     # kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
#                 except:
#                     kpt.append(np.zeros((17,3)))
#                     kptScore.append(np.zeros(17))
#             poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
#             # print(np.array(kpt).shape)
#     t4_pose = time.time()


#     print(f"input : {imgTensors.shape} ")
#     print(f'''
#     POSE total  cost : {round(t4_pose-t1_pose,4)} sec
#     preprocess  cost : {round(t2_pose-t1_pose,4)} sec
#     inference   cost : {round(t3_pose-t2_pose,4)} sec
#     tracking    cost : {round(t4_pose-t3_pose,4)} sec
#             ''')
            
#     count += 1
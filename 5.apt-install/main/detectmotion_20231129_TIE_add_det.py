from cfg_testInteractive import *
from utils import * 
sourcesNum = len(video_list)

from Object_PoseEstimate_v8 import PoseEstimate,ObjDetect
from Object_send_server import DetectSendor
from modules.MotionTransform import CustomTransform

# import torch
# from torchvision.transforms import Compose

# import cv2
# import numpy as np
# import json
from pyskl.apis import inference_recognizer, init_recognizer
import mmcv

import queue, threading
# import datetime,time


pose_last_shown = time.time()


# pose estimate
PoseEstimator = PoseEstimate(device = f"cuda:{device_use_Pose}",
                            engine_file_path=f"weights/yolov8l-pose.pt")

ObjDetector = ObjDetect(device = f"cuda:{device_use_Pose}",
                            engine_file_path=f"weights/yolov8l.pt")



track_history = [] ; drop_counting = [] ; write2SmbLast = []
for _ in range(sourcesNum):
    track_history.append(defaultdict(lambda: []))
    drop_counting.append(defaultdict(lambda: 0))
    write2SmbLast.append(0)

clss = []
f = open(clss_path)
for line in f.readlines():
    clss.append(line[:-1])
f.close()

config = mmcv.Config.fromfile(stgcnpp_config)
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
stgcnpp_model = init_recognizer(config, stgcnpp_checkpoint, f"cuda:{device_use_motion}").half()
stgcnpp_model.eval()  # 切换到推理模式
previous_t3 = time.time()


def main():

    for i in range(10):
        t1_warm = time.time()
        with torch.no_grad():
            transfered_set = torch.empty((10,10,2,100,17,3), dtype=torch.float32).half().to(f"cuda:{device_use_motion}")
            result = stgcnpp_model(keypoint = transfered_set)
        t2_warm = time.time()
    print(f"Finished STGCNPP warm-up ......")

    global pose_last_shown
    global imgTensors
    global poseTrackResults
    global GFMotion ; global ThMotion
    lastInferenceMotion = time.time()

    motionQ = queue.Queue()
    tM = threading.Thread(target = process_content_Motion, args = (motionQ,))
    tM.setDaemon(True)
    tM.start()

    vidcaps = multi_video_load(video_list)
    imgs = [np.zeros((model_trained_scale_h,model_trained_scale_w,3)).astype('uint8')] * sourcesNum
    poseTrackResults = [None]*sourcesNum
    ThMotion = False
    GFMotion = False
    countFrame = 0
    while True:
        detResult = None ; detResult_info = None
        t0_pose = time.time()
        imgs = sources_read(vidcaps,video_list,imgs)
        t1_pose = time.time()
        outputs = PoseEstimator.poseTrack(imgs = imgs)
        t2_pose = time.time()

        if aid_det and torch.is_tensor(outputs[-1]):
            detResult = ObjDetector.detPred(imgs = outputs[-1],
                                            classes=cls_det,half=True)


        for lineIndex,output in enumerate(outputs[:-1]):
            poseTrackResult = [] ; track_ids_conform_frame_num = [] ; boxesSort = []

            track_ids = output.boxes.id
            if track_ids is None:
                track_ids = []
            else:
                track_ids = track_ids.int().cpu().tolist()
            boxes = output.boxes.xywh.cpu().tolist()
            keypoints = output.keypoints.data

            diff = list(set(list(set(track_history[lineIndex].keys()))).difference(track_ids))
            for d in diff:
                if drop_counting[lineIndex][d] > max_miss:
                    del drop_counting[lineIndex][d]
                    del track_history[lineIndex][d]
                else:
                    drop_counting[lineIndex][d]+=1
            
            for box, track_id,keypoint in zip(boxes, track_ids,keypoints):
                track = track_history[lineIndex][track_id]
                track.append(keypoint.unsqueeze(0))
                if len(track) > kptSeqNum:  # retain 90 tracks for 90 frames
                    track.pop(0)
                if len(track) == kptSeqNum:
                    poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
                    track_ids_conform_frame_num.append(track_id)
                    boxesSort.append(box)
            poseTrackResults[lineIndex] = [poseTrackResult,track_ids_conform_frame_num,boxesSort]
        t3_pose = time.time()

        if not shutthefuckup:
            cu = time.time()
            if cu - pose_last_shown >= pose_shown_interval:
                print(f'''[POSE] 
        capture cost : {round(t1_pose-t0_pose,4)} sec
        POSE total  cost : {round(t3_pose-t1_pose,4)} sec
        ''')
                if detResult is not None:
                    print("detection successful --> len(detResult) : ",len(detResult))
                pose_last_shown = cu

        if ThMotion == False and poseTrackResults != [None]*sourcesNum and countFrame>= kptSeqNum and time.time()-lastInferenceMotion > motionInferenceInterval:
            motionQ.put({"cmd"           : "invoke",
                         "poseTrackResults" :  poseTrackResults,
                         "imgs" : imgs})
            lastInferenceMotion = time.time()
        if GFMotion == True:
            ThMotion = False
            poseTrackResults = [None]*sourcesNum
        countFrame+= 1
        if t3_pose-t0_pose <= 0.03:
            time.sleep(0.03-t3_pose+t0_pose) # 30 fps (33ms)

def process_content_Motion(queue):
    global GFMotion ; global ThMotion
    global previous_t3
    while True:
        if (queue.empty()):
            time.sleep(0.05)  
        else:
            msg = queue.get()
            cmd = msg.get('cmd')
            poseTrackResults = msg.get('poseTrackResults')
            imgs = msg.get('imgs')
            if cmd == 'quit':
                print("收到 quit 訊息，即將結束")
                break      
            if cmd == 'invoke':
                SmbPath = None
                current_time = time.time()
                current_datetime =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                current_date = str(datetime.date.today())
                belongLine = [] ; bboxesSorts = [] ; trackIDss = []    
                keypointsSeqs_transformed = []       
                       
                t1_motion = time.time()  
                for lineIndex in range(len(poseTrackResults)):
                    poseTrackResult = poseTrackResults[lineIndex]
                    trackPersonNum = len(poseTrackResult[0])
                    
                    eventDicts = [] 
                    if trackPersonNum:
                        keypointsSeqs = poseTrackResult[0] # list of many (1,40,17,3)
                        trackIDs = poseTrackResult[1]
                        bboxesSort = poseTrackResult[2]                        

                        tpre1 = time.time()
                        keypointsSeqs = torch.cat(keypointsSeqs).unsqueeze(1)#.reshape((-1,1,kptSeqNum, 17,3))
                        keypointsSeqsORG = torch.clone(keypointsSeqs)
                        keypointsSeqs[...,0] = keypointsSeqs[...,0]*(model_trained_scale_w/video_w)
                        keypointsSeqs[...,1] = keypointsSeqs[...,1]*(model_trained_scale_h/video_h)

                        for kptIndex,keypointsSeq in enumerate(keypointsSeqs):
                            if ROI_all_cams[lineIndex] is not None:
                                foot_kpt_last = keypointsSeqsORG[kptIndex,0,-1,13:,:] 
                                min_score_of_foot = torch.min(foot_kpt_last[...,-1])
                                if min_score_of_foot < foot_visibility_thres:
                                    continue

                                bbox_of_foot = torch.tensor([torch.min(foot_kpt_last[:,0]),torch.min(foot_kpt_last[:,1]),torch.max(foot_kpt_last[:,0]),torch.max(foot_kpt_last[:,1])]).tolist()
                                x1,y1,x2,y2 = bbox_of_foot
                                bbox_of_foot = box(x1,y1,x2,y2)
                                overlap_orNot = ROI_all_cams[lineIndex].intersects(bbox_of_foot)
                                print("overlap_orNot : ",overlap_orNot)
                                if overlap_orNot:
                                    keypointsSeqs_transformed.append(transform(keypointsSeq))#(1,40,17,3))
                                    belongLine.append(lineIndex)
                                    trackIDss.append(trackIDs[kptIndex])
                                    bboxesSorts.append(bboxesSort[kptIndex])

                if len(keypointsSeqs_transformed):
                    keypointsSeqs_transformed = torch.cat(keypointsSeqs_transformed, dim=0)#.to(f'cuda:{device_use_motion}')
                    t2_motion = time.time()                    
                    keypointsSeqs_transformed = keypointsSeqs_transformed.to(f'cuda:{device_use_motion}').half()
                    t3_motion = time.time() 

                    with torch.no_grad():
                        results = stgcnpp_model(keypoint = keypointsSeqs_transformed)#
                    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    t4_motion = time.time() 
                    results = results.cpu().numpy()
                    actIndexes,actScores,top10_indices_sorted,top10_values_sorted = motion_result_postprocess(results)
                    t5_motion = time.time() 
                    if not shutthefuckup:
                        print(f'''\n{symbo*80}\nMotion inference :
        * Total      cost : {round(t5_motion-t1_motion,4)} sec 
        *        tfm cost : {round(t2_motion-t1_motion,4)} sec 
        * device transfer : {round(t3_motion-t2_motion,4)} sec 
        * inference  cost : {round(t4_motion-t3_motion,4)} sec 
        * device tsf cost : {round(t5_motion-t4_motion,4)} sec 
        * Total people Num : {len(results)}
                                ''')

                    if write2Smb:
                        eventDicts = rawDetect2Info(actIndexes=actIndexes,actScores=actScores,
                                                    keypointsSeqs=keypointsSeqs,bboxesSort=np.array(bboxesSort),
                                                    belongLine=belongLine,trackIDss=np.array(trackIDss),
                                                    threshold_Neighbor=threshold_Neighbor,imgs=imgs,
                                                    writeImg=True)
                        try:
                            if len(eventDicts):
                                print("eventDicts : " , len(eventDicts))
                                print("sending (Event Server)..............................")
                                detectSendor.sendEvents(eventDicts)
                                t2_send = time.time()
                                print("successful Sending (Event Server)..........................................\nTake time :",t2_send-t1_send)
                            else:
                                print("No event")
                        except:
                            print("Send Server fail......................................................")

                    GFMotion = True
                    ThMotion = True


transform = CustomTransform()

ROI_all_cams = ROIs_define_showUse(video_list)
print(f"{symbo*60}\nvideo_list : ",video_list)
print(f"tROI_all_cams:\n",ROI_all_cams)
# exit()
detectSendor = DetectSendor()
detectSendor_pred = DetectSendor(qName='pred_queue')

if __name__ == "__main__":
    main()
shotthefuckup = True
alarm_event_list = [0,1,2,3,4,5,6,7,9,10]
PUBLIC_THRESHOLD = 80
write2Smb = True
write2SmbInterval = 2 # second

video_list = ['rtsp://192.168.8.222:7070/stream2']*1+['rtsp://192.168.8.166:7070/stream2']*1
sourcesNum = len(video_list)

device_use_Pose = 0
device_use_motion = 1
kptSeqNum = 40
max_miss=5
motionInferenceInterval = 0.5

sendEventServer = True
cycle = True # video mode cycle
cuda_cv = True
symbo = "*"
from Object_PoseEstimate_v8 import PoseEstimate
from Object_send_server import DetectSendor
from modules import VideoCaptureHard as cap

import time
import torch
from torchvision.transforms import Compose

import cv2
import numpy as np
import os
import json
from pyskl.apis import inference_recognizer, init_recognizer
import mmcv

import queue
import threading
from collections import defaultdict
import datetime
from shapely.geometry import Polygon, box

# -------------------------------------------------------------------------------
'''
0 : sit down
1 : stand up
2 : reach into pocket
3 : rub hands
4 : staggering
5 : sitting
6 : standing
7 : lie down
8 : cross foot to straight
9 : walking  
'''
actionMatch2ORGcls=defaultdict(lambda: [])
actionMatch2ORGcls[0] = 7
actionMatch2ORGcls[1] = 8
actionMatch2ORGcls[2] = 126#24
actionMatch2ORGcls[3] = 125#33
actionMatch2ORGcls[4] = 124#41
actionMatch2ORGcls[5] = 120
actionMatch2ORGcls[6] = 121
actionMatch2ORGcls[7] = 122
actionMatch2ORGcls[10] = 64
actionMatch2ORGcls[9] = 127

# actionMatch2ORGcls[8] = 8787
# actionMatch2ORGcls[9] = 7878

# -------------------------------------------------------------------------------
# pose estimate
PoseEstimator = PoseEstimate(device = f"cuda:{device_use_Pose}",
                            engine_file_path=f"weights/yolov8s-pose.pt")

track_history = []
drop_counting = []
write2SmbLast = []
for _ in range(sourcesNum):
    track_history.append(defaultdict(lambda: []))
    drop_counting.append(defaultdict(lambda: 0))
    write2SmbLast.append(0)
# -------------------------------------------------------------------------------
# motion classification
# stgcnpp_checkpoint = "./weights/yolov8_stgcnpp_123_20230901.pth" # train with yolov8pose
# stgcnpp_config = "./configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"

# -------------------------------------------------------------------------------
# stgcnpp_checkpoint = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_10cls_for_show/j/best_top1_acc_epoch_13.pth" # train with yolov8pose
# stgcnpp_config = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_10cls_for_show/j/j.py"
# clss_path = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_10cls_for_show/j/cls.txt"

# -------------------------------------------------------------------------------
stgcnpp_checkpoint = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_11cls_for_show_20231002/j/yolov8_stgcnpp_SHOWUSE_cls_11_20231002.pth" # train with yolov8pose
stgcnpp_config = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_11cls_for_show_20231002/j/j.py"
clss_path = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_11cls_for_show_20231002/j/cls.txt"

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
        # print(round(t2_warm-t1_warm,4))
    print(f"Finished STGCNPP warm-up ......")

    global imgTensors
    global poseTrackResults
    global GFMotion ; global ThMotion
    lastInferenceMotion = time.time()

    motionQ = queue.Queue()
    tM = threading.Thread(target = process_content_Motion, args = (motionQ,))
    tM.setDaemon(True)
    tM.start()

    vidcaps = multi_video_load(video_list)
    imgs = [np.zeros((1080,1920,3)).astype('uint8')] * sourcesNum
    poseTrackResults = [None]*sourcesNum
    ThMotion = False
    GFMotion = False
    countFrame = 0
    while True:
        t0_pose = time.time()
        imgs = sources_read(vidcaps,video_list,imgs)

        t1_pose = time.time()
        outputs = PoseEstimator.poseTrack(imgs = imgs)
        t2_pose = time.time()
        for lineIndex,output in enumerate(outputs):
            poseTrackResult = [] ; track_ids_conform_frame_num = [] ; boxesSort = []

            track_ids = output.boxes.id#.int().cpu().tolist()
            if track_ids is None:
                track_ids = []
            else:
                track_ids = track_ids.int().cpu().tolist()
            boxes = output.boxes.xywh.cpu().tolist()
            keypoints = output.keypoints.data#.cpu().numpy()

            diff = list(set(list(set(track_history[lineIndex].keys()))).difference(track_ids))
            for d in diff:
                if drop_counting[lineIndex][d] > max_miss:
                    del drop_counting[lineIndex][d]
                    del track_history[lineIndex][d]
                    # print(f"{symbo*80}\nDrop id : {d} from line {lineIndex}")
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

        if not shotthefuckup:
            print(f'''[POSE] 
        capture cost : {round(t1_pose-t0_pose,4)} sec
        POSE total  cost : {round(t3_pose-t1_pose,4)} sec
        ''')
        #    inference   cost : {round(t2_pose-t1_pose,4)} sec
        # postprocess cost : {round(t3_pose-t2_pose,4)} sec
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

def multi_video_load(sourcesList):
    vidcaps = []
    for video_index,source in enumerate(sourcesList):
        if "rtsp" in video_list[video_index]:
            # vidcaps.append(cap.VideoCapture(video_list[video_index]))
            vidcaps.append(cap.VideoCapture(source,0,0,0))

        else:
            vidcaps.append(cv2.VideoCapture(source))
    return vidcaps

def sources_read(vidcaps,sourcesList,imgs):
    for video_index,source in enumerate(sourcesList):
        ret, image = vidcaps[video_index].read()
        if not ret and cycle:
            vidcaps[video_index].release()
            time.sleep(0.001)
            if "rtsp" in video_list[video_index]:
                # vidcaps[video_index] = cap.VideoCapture(video_list[video_index])
                vidcaps[video_index] = cap.VideoCapture(video_list[video_index],3840,2160,0)
            else:
                vidcaps[video_index] = cv2.VideoCapture(video_list[video_index])
            continue
        elif not ret and not cycle:
            print("video Finished ...............")
        else:
            imgs[video_index] = image
            # imgs[video_index] = cv2.flip(image,0)
    return imgs

def ROIs_define_showUse(video_list):
    infos = {}
    for video_name in video_list:
        infos[video_name] = None

    f = open("ROI_setup/ROI_show_use.txt")
    for line in f.readlines():
        info = json.loads(line[:-1])
        rtsp_info = info['RTSP']
        poly_info = info['poly']
        if rtsp_info in infos.keys():
            infos[rtsp_info] = poly_info
    f.close()
    ROI_all_cams = []
    for video_name in video_list:
        if  video_name in infos.keys():
            ROI_all_cams.append(Polygon(infos[video_name]))
            
        else:
            ROI_all_cams.append(None)
    return ROI_all_cams


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
                belongLine = []
                t1_motion = time.time()  
                keypointsSeqs_transformed = []       
                trackIDss = []           
                bboxesSorts = []
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
                        keypointsSeqs[...,0] = keypointsSeqs[...,0]*(1920/1280)
                        keypointsSeqs[...,1] = keypointsSeqs[...,1]*(1280/720)

                        for kptIndex,keypointsSeq in enumerate(keypointsSeqs):
                            # x,y,w,h = bboxesSort[kptIndex]
                            # x1,y1 = x-w/2,y-h/2
                            # x2,y2 = x+w/2,y+h/2
                            # if x1 < 10 or x2 > 1270 or y1 < 10 or y2 > 710:
                            #     print(f"bbox [{x1},{y1},{x2},{y2}] Not in effective range so skip.......")
                            #     continue
                            ############################################################################
                            ### ROI new ######################################
                            if ROI_all_cams[lineIndex] is not None:
                                # print(keypointsSeq.shape)# (40 ,17, 3)
                                foot_kpt_last = keypointsSeqsORG[kptIndex,0,-1,13:,:] 
                                min_score_of_foot = torch.min(foot_kpt_last[...,-1])
                                if min_score_of_foot < 0.2:
                                    continue
                                # print("min_score_of_foot : ",min_score_of_foot)
                                # print(foot_kpt_last.shape)
                                bbox_of_foot = torch.tensor([torch.min(foot_kpt_last[:,0]),torch.min(foot_kpt_last[:,1]),torch.max(foot_kpt_last[:,0]),torch.max(foot_kpt_last[:,1])]).tolist()
                                x1,y1,x2,y2 = bbox_of_foot
                                # print(x1,y1,x2,y2)
                                bbox_of_foot = box(x1,y1,x2,y2)
                                # print(bbox_of_foot)
                                overlap_orNot = ROI_all_cams[lineIndex].intersects(bbox_of_foot)
                                print("overlap_orNot : ",overlap_orNot)
                                if overlap_orNot:
                                    keypointsSeqs_transformed.append(transform(keypointsSeq))#(1,40,17,3))
                                    belongLine.append(lineIndex)
                                    trackIDss.append(trackIDs[kptIndex])
                                    bboxesSorts.append(bboxesSort[kptIndex])
                            # keypointsSeqs_transformed.append(transform(keypointsSeq))#(1,40,17,3))
                            # belongLine.append(lineIndex)
                            # trackIDss.append(trackIDs[kptIndex])
                            # bboxesSorts.append(bboxesSort[kptIndex])
                if len(keypointsSeqs_transformed):
                    keypointsSeqs_transformed = torch.cat(keypointsSeqs_transformed, dim=0)#.to(f'cuda:{device_use_motion}')
                    t2_motion = time.time()                    
                    keypointsSeqs_transformed = keypointsSeqs_transformed.to(f'cuda:{device_use_motion}').half()
                    t3_motion = time.time() 

                    with torch.no_grad():
                        results = stgcnpp_model(keypoint = keypointsSeqs_transformed)#
                    
                    t4_motion = time.time() 
                    results = results.cpu().numpy()
                    actIndexes,actScores = np.argmax(results,axis=1),np.max(results,axis=1)
                    actScores = (100*actScores).astype(int)

                    top10_indices = np.argpartition(-results, 9, axis=1)[:, :9]
                    top10_values = np.array([row[row_indices] for row, row_indices in zip(results, top10_indices)])
                    sorted_order = np.argsort(-top10_values, axis=1)
                    top10_values_sorted = [row[row_order] for row, row_order in zip(top10_values, sorted_order)]
                    top10_indices_sorted = [row_indices[row_order] for row_indices, row_order in zip(top10_indices, sorted_order)]
                    t5_motion = time.time() 
                    if not shotthefuckup:
                        print(f'''\n{symbo*80}\nMotion inference :
        * Total      cost : {round(t5_motion-t1_motion,4)} sec 
        *        tfm cost : {round(t2_motion-t1_motion,4)} sec 
        * device transfer : {round(t3_motion-t2_motion,4)} sec 
        * inference  cost : {round(t4_motion-t3_motion,4)} sec 
        * device tsf cost : {round(t5_motion-t4_motion,4)} sec 
        * Total people Num : {len(results)}
                                ''')
                    #        * data distribute : {belongLine}
                    current = time.time()
                    for resultID,(actIndex,actScore,belongLineIndex) in enumerate(zip(actIndexes,actScores,belongLine)):
                        if actIndex in alarm_event_list and actScore >= PUBLIC_THRESHOLD:
                            ctIndex = actionMatch2ORGcls[actIndex] # act match to old cls
                            if write2Smb and sendEventServer:
                                imgName = f"/home/samba/raw_result/{current_date}/camIndex_{belongLineIndex}/camIndex_{belongLineIndex}_{current_datetime}_ACT_{actIndex}_{actScore}.png"
                                SmbPath = imgName.replace("/home/samba/raw_result","/images")
                                subPath = os.path.dirname(imgName)
                                subsubPath = os.path.dirname(subPath)
                                if not os.path.exists(subsubPath):
                                    os.mkdir(subsubPath)
                                if not os.path.exists(subPath):
                                    os.mkdir(subPath)

                                bbox =  bboxesSorts[resultID]
                                w_half,h_half = bbox[2]/2,bbox[3]/2
                                p1 = (int(bbox[0]-w_half),int(bbox[1]-h_half))
                                p2 = (int(bbox[0]+w_half),int(bbox[1]+h_half))
                                write_img = imgs[belongLineIndex].copy()
                                cv2.rectangle(write_img,p1,p2,(0,0,255),1)
                                write_img = cv2.resize(write_img,(720,540))
                                cv2.imwrite(imgName,write_img)
                                print("Write : ", imgName)

                                x = bboxesSorts[resultID][0]#+(bboxesSorts[resultID][2]/2)
                                y = bboxesSorts[resultID][1]#+(bboxesSorts[resultID][3]/2)
                                x = x*(3840/1280)
                                y = y*(2160/720)
                                eventDicts.append({
                                                    "start_time":0, "user_id":9487,
                                                    "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(trackIDss[resultID]).zfill(7)}",
                                                    "event_action": int(actIndex),
                                                    "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                                                    "confidence":int(actScore),"prediction_status":0,
                                                    "event_action_id_2nd": 0,"confidence_2nd":0,"event_action_id_3rd": 0,"confidence_3rd":0,
                                                    "snapshot":SmbPath,
                                                    "center_x":int(x),
                                                    "center_y": int(y),
                                                    })


                    if len(eventDicts):
                        try:
                            print(f"[{current}] sending (Event Server) :{len(eventDicts)}")
                            detectSendor.sendEvents(eventDicts)
                            print("successful Sending (Event Server)..........................................\nTake time :",t2_send-t1_send)
                        except:
                            print("Send Server fail......................................................")

                    GFMotion = True
                    ThMotion = True


class CustomTransform(torch.nn.Module):
    def __init__(self):
        super(CustomTransform, self).__init__()
        self.num_clips = 10
        self.clip_len = 100
        self.threshold = 0.01
        self.w = 1920
        self.h = 1080
        self.num_person = 2
        self.M = 2
        self.nc = 10
        self.seqNum = 1000//10
        self.V = 17
        self.C = 3

    def forward(self, keypoint):
        mask = keypoint[..., 2] > self.threshold
        maskout = keypoint[..., 2] <= self.threshold
        keypoint[..., 0] = (keypoint[..., 0] - (self.w / 2)) / (self.w / 2)
        keypoint[..., 1] = (keypoint[..., 1] - (self.h / 2)) / (self.h / 2)
        keypoint[..., 0][maskout] = 0
        keypoint[..., 1][maskout] = 0
        num_frames = keypoint.shape[1]

        allinds = []
        for _ in range(self.num_clips):
            start = torch.randint(0, num_frames, (1,))
            inds = torch.arange(int(start), int(start) + self.clip_len)
            allinds.append(inds)
        inds = torch.cat(allinds)

        inds = inds % num_frames
        transitional = torch.zeros(num_frames, dtype=torch.bool)
        inds_int = inds.long()
        coeff = transitional[inds_int]
        coeff = coeff.int()
        inds = (coeff * inds_int + (1 - coeff) * inds).float()

        keypoint = keypoint[:, inds.long()].float()

        pad_dim = self.num_person - keypoint.shape[0]
        pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype) # device = f"cuda:{device_use_motion}"
        # pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype,device = keypoint.get_device())

        keypoint = torch.cat((keypoint, pad), dim=0)
        keypoint = keypoint.view(self.M, self.nc, self.seqNum, self.V, self.C)
        keypoint = keypoint.transpose(1, 0)
        keypoint = keypoint.contiguous()
        keypoint = torch.unsqueeze(keypoint, 0)

        return keypoint

transform = CustomTransform()

ROI_all_cams = ROIs_define_showUse(video_list)
print(f"{symbo*60}\nvideo_list : ",video_list)
print(f"tROI_all_cams:\n",ROI_all_cams)
# exit()
detectSendor = DetectSendor()
detectSendor_pred = DetectSendor(qName='pred_queue')

if __name__ == "__main__":
    main()
    
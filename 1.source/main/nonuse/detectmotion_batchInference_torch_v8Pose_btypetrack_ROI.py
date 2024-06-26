'''
[TODO]
* preprocess (mats>np.hstack>normalize>to.device>Torch Resize>track)
    - cv2.gpuMat keep

'''

shotthefuckup = False
alarm_event_list = [6,21,24,26,42,64,79,92,101,122]
# alarm_event_list = list(range(123)) # for test smb
PUBLIC_THRESHOLD = 80
write2Smb = True
write2SmbInterval = 10 # second


# video_list = ['rtsp://10.1.1.31:7070/stream2']*6+['rtsp://10.1.1.32:7070/stream2']*6
video_list = ['rtsp://10.1.1.31:7070/stream2']*6+['rtsp://10.1.1.32:7070/stream2']*6

# video_list = [
#                 "rtsp://admin:1qaz!QAZ@192.168.1.21:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.22:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.23:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.24:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.25:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.26:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.27:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.28:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.29:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.30:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.31:7070/stream2",
#                 "rtsp://admin:1qaz!QAZ@192.168.1.32:7070/stream2"
# ]
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

track_history = []
drop_counting = []
write2SmbLast = []
for _ in range(sourcesNum):
    track_history.append(defaultdict(lambda: []))
    drop_counting.append(defaultdict(lambda: 0))
    write2SmbLast.append(0)
# -------------------------------------------------------------------------------
# motion classification
stgcnpp_checkpoint = "./weights/yolov8_stgcnpp_123_20230901.pth" # train with yolov8pose
stgcnpp_config = "./configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"
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
    imgs = [np.zeros((2160,3840,3)).astype('uint8')] * sourcesNum
    poseTrackResults = [None]*sourcesNum
    ThMotion = False
    GFMotion = False
    countFrame = 0
    while True:
        t0_pose = time.time()
        imgs = sources_read(vidcaps,video_list,imgs)
        # for imi,img in enumerate(imgs):
            # print(imi,img.shape)
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

            # annotated_frame = result.plot()
            diff = list(set(list(set(track_history[lineIndex].keys()))).difference(track_ids))
            for d in diff:
                if drop_counting[lineIndex][d] > max_miss:
                    del drop_counting[lineIndex][d]
                    del track_history[lineIndex][d]
                    # print(f"{symbo*80}\nDrop id : {d} from line {lineIndex}")
                else:
                    drop_counting[lineIndex][d]+=1
            
            for box, track_id,keypoint in zip(boxes, track_ids,keypoints):
                # x, y, w, h = box
                track = track_history[lineIndex][track_id]
                track.append(keypoint.unsqueeze(0))
                # track.append(keypoint)
                if len(track) > kptSeqNum:  # retain 90 tracks for 90 frames
                    track.pop(0)
                if len(track) == kptSeqNum:
                    # poseTrackResult.append(torch.hstack(track).reshape((-1, 17,3)).cpu())
                    poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
                    track_ids_conform_frame_num.append(track_id)
                    boxesSort.append(box)
                # print(f"{track_id}({lineIndex}):{kpt_seq.shape}")
            poseTrackResults[lineIndex] = [poseTrackResult,track_ids_conform_frame_num,boxesSort]
        t3_pose = time.time()

        # print(f"input : {imgTensors.shape} output : {outputs.shape}")
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
    return imgs

def ROIs_define():
    f = open("ROI_setup/cams_Sort.txt")
    camsSort = []
    for line in f.readlines():
        camsSort.append(line[:-1])
    f.close()

    ROI_all_cams = []
    for cam in camsSort:
        if os.path.exists("ROI_setup/"+cam):
            f = open("ROI_setup/"+cam)
            ROI_EachCam = []
            for line in f.readlines():
                x,y,w,h = line.split(" ")
                h = h[:-1]
                x,y,w,h = int(x),int(y),int(w),int(h)
                ROI_EachCam.append([x,y,w,h])
            f.close()
            if len(ROI_EachCam):
                ROI_all_cams.append(ROI_EachCam)
            else:
                ROI_all_cams.append(None)
        else:
            print("ROI_setup/"+cam, " not exist....")
            ROI_all_cams.append(None)
    return ROI_all_cams

def is_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True


# def posePreprocessV1(lineIndex,img):
#     global imgTensors
#     imgTensor = imgs2Tensor(img)
#     imgTensors[lineIndex] = imgTensor

# def posePreprocessV2(img):
#     # global imgTensors
#     t1_resize = time.time()
#     img = Resize(img,(960,576),False)/255
#     # t2_resize = time.time()
#     # print("t2_resize-t1_resize",t2_resize-t1_resize)
#     img = img.transpose(2, 0, 1)
#     return torch.from_numpy(img)#.to(f"cuda:{device_use_Pose}").half()

# def Resize(img_cpu,sizeNew,cuda_cv = False):
#     if cuda_cv:
#         try:
#             gpu_frame = cv2.cuda_GpuMat()
#             gpu_frame.upload(img_cpu)
#             gpu_frame = cv2.cuda.resize(gpu_frame,sizeNew)
#             img_cpu = gpu_frame.download()
#         except:
#             # img_cpu = cv2.resize(img_cpu,sizeNew)
#             img_cpu = cv2.resize(img_cpu,sizeNew, interpolation = cv2.INTER_LINEAR)

#     else:
#         img_cpu = cv2.resize(img_cpu,sizeNew)
#     return img_cpu

# def motionPreprocessor():
#     global poseTrackResults
#     for lineIndex,poseTrackResult in enumerate(poseTrackResults):
#         poseTrackResult[trackIndex]["keypoints_seq"]
#         poseTrackResult[trackIndex]["keypointScore_seq"]

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
                    
                    eventDicts = [] ; predDicts = []
                    if trackPersonNum:
                        bboxesSort = poseTrackResult[2]
                        bboxesSorts+=bboxesSort
                        ROI_eachCam = ROI_all_cams[lineIndex] 
                        tpre1 = time.time()
                        keypointsSeqs = torch.cat(poseTrackResult[0]).unsqueeze(1)#.reshape((-1,1,kptSeqNum, 17,3))
                        # print(keypointsSeqs.shape,"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
                        keypointsSeqs[...,0] = keypointsSeqs[...,0]*(1920/1280)
                        keypointsSeqs[...,1] = keypointsSeqs[...,1]*(1280/720)

                        trackIDs = poseTrackResult[1]
                        for kptIndex,keypointsSeq in enumerate(keypointsSeqs):
                            # if ROI_eachCam is not None:
                                ### -------------------------------------------------------------------------------------#
                                ### < ROI >
                                # print(keypointsSeq.shape)# (40 ,17, 3)
                                # foot_kpt_last = keypointsSeq[0,-1,13:,:] 
                                # bbox_of_foot = torch.tensor([torch.min(foot_kpt_last[:,0]),torch.min(foot_kpt_last[:,1]),torch.max(foot_kpt_last[:,0])-torch.min(foot_kpt_last[:,0]),torch.max(foot_kpt_last[:,1])-torch.min(foot_kpt_last[:,1])])
                                # if is_overlap(bbox_of_foot,ROI_eachCam[0]):
                                #     # ROI_color = (0,0,255)
                                #     # cv2.putText(image_, f"ROI intrusion!", (int(x), int(y-40)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                #     eventDicts.append({
                                #                         "start_time":current_time, "user_id":9487,
                                #                         "uu_id":f"{str(lineIndex).zfill(2)}{str(trackIDs[kptIndex]).zfill(7)}","event_action": 998,
                                #                         "status":5,"group_id":'G01',"location_id":lineIndex,
                                #                         "confidence":100,"prediction_status":0,
                                #                         "event_action_id_2nd": 0,"confidence_2nd":0,"event_action_id_3rd": 0,"confidence_3rd":0,
                                #                         "snapshot":'/home/samba/raw_result/you9527.png',"center_x":0,"center_y":0 
                                #                         })
                            keypointsSeqs_transformed.append(transform(keypointsSeq))#(1,40,17,3))
                            belongLine.append(lineIndex)
                            trackIDss.append(trackIDs[kptIndex])
                            # print(f"(motion Q ({lineIndex})) after : ", keypointsSeq.shape)

                if len(keypointsSeqs_transformed):
                    keypointsSeqs_transformed = torch.cat(keypointsSeqs_transformed, dim=0)#.to(f'cuda:{device_use_motion}')
                    t2_motion = time.time()                    
                    keypointsSeqs_transformed = keypointsSeqs_transformed.to(f'cuda:{device_use_motion}').half()
                    t3_motion = time.time() 

                    # print("before inference",keypointsSeqs_transformed.shape,keypointsSeqs_transformed.get_device())
                    # print("(motion Q) trackIDs : ",trackIDs)
                    # print("(motion Q) preprocess time : ",t2_motion-t1_motion, " (sec)")
                    with torch.no_grad():
                        results = stgcnpp_model(keypoint = keypointsSeqs_transformed)#
                    
                    # results= torch.max(results,dim=1)
                    t4_motion = time.time() 
                    # actIndexes,actScores = results.indices.cpu().tolist(),results.values.cpu().numpy()
                    results = results.cpu().numpy()
                    actIndexes,actScores = np.argmax(results,axis=1),np.max(results,axis=1)
                    actScores = (100*actScores).astype(int)
                    # print("actIndexes : ",actIndexes)
                    # print("actScores : ",actScores)
                    top10_indices = np.argpartition(-results, 10, axis=1)[:, :10]
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

                    SmbPaths = [None]*sourcesNum
                    current = time.time()
                    # have_pic = []
                    for resultID,(actIndex,actScore,belongLineIndex) in enumerate(zip(actIndexes,actScores,belongLine)):
                        if actIndex in alarm_event_list and actScore >= PUBLIC_THRESHOLD and SmbPaths[belongLineIndex] is None:
                            
                            if write2Smb and current-write2SmbLast[belongLineIndex] >= write2SmbInterval:
                                
                                imgName = f"/home/samba/raw_result/{current_date}/camIndex_{belongLineIndex}/camIndex_{belongLineIndex}_{current_datetime}_ACT_{actIndex}_{actScore}.png"
                                SmbPaths[belongLineIndex] = imgName.replace("/home/samba/raw_result","/images")
                                subPath = os.path.dirname(imgName)
                                subsubPath = os.path.dirname(subPath)
                                if not os.path.exists(subsubPath):
                                    os.mkdir(subsubPath)
                                    
                                if not os.path.exists(subPath):
                                    os.mkdir(subPath)

                                cv2.imwrite(imgName,imgs[belongLineIndex])
                                # have_pic.append(True)
                                print("Write : ", imgName)
                                write2SmbLast[belongLineIndex] = current
                            # else:
                            #    have_pic.append(False)
                    if sendEventServer:
                        try:
                            t1_send = time.time()         
                            for resultID,(actIndex,actScore,belongLineIndex) in enumerate(zip(actIndexes,actScores,belongLine)):
                                # print(resultID,actIndex,type(actScore))
                                if actScore > PUBLIC_THRESHOLD:                             
                                    shapshotPath = SmbPaths[belongLineIndex]
                                    # print(actIndex,actScore)
                                    if shapshotPath is None:
                                        shapshotPath = "None"
                                    else:
                                        eventDicts.append({
                                                            "start_time":0, "user_id":9487,
                                                            "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(trackIDss[resultID]).zfill(7)}",
                                                            "event_action": int(actIndex),
                                                            "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                                                            "confidence":int(actScore),"prediction_status":0,
                                                            "event_action_id_2nd": 0,"confidence_2nd":0,"event_action_id_3rd": 0,"confidence_3rd":0,
                                                            "snapshot":shapshotPath,
                                                            "center_x":int(bboxesSorts[resultID][0]+(bboxesSorts[resultID][2]/2)),
                                                            "center_y": int(bboxesSorts[resultID][1]+(bboxesSorts[resultID][3]/2)),
                                                            })

                                predDicts.append({
                                                    "start_time":0, "user_id":9487,
                                                    "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(trackIDss[resultID]).zfill(7)}",
                                                    "event_action": int(actIndex),
                                                    "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                                                    "confidence":int(actScore),"prediction_status":0,
                                                    "event_action_id_2nd": 0,"confidence_2nd":0,"event_action_id_3rd": 0,"confidence_3rd":0,
                                                    "snapshot":shapshotPath,
                                                    "center_x":int(bboxesSorts[resultID][0]+(bboxesSorts[resultID][2]/2)),
                                                    "center_y": int(bboxesSorts[resultID][1]+(bboxesSorts[resultID][3]/2)),
                                                    "actResults":np.column_stack((top10_indices_sorted[resultID], top10_values_sorted[resultID])).tolist(),
                                                    })
                            if len(eventDicts):
                                print(len(eventDicts))
                                print("sending (Event Server)..............................")
                                detectSendor.sendEvents(eventDicts)
                                t2_send = time.time()
                                print("successful Sending (Event Server)..........................................\nTake time :",t2_send-t1_send)
                            if len(predDicts):
                                print("sending (Pred module)..............................")
                                detectSendor_pred.sendEvents(predDicts)
                                # print(predDicts[0]["actResults"])
                                print("successful Sending (Pred module)..........................................")

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

ROI_all_cams = ROIs_define()
print(f"{symbo*60}\n\tROI_all_cams:\n",ROI_all_cams)

detectSendor = DetectSendor()
detectSendor_pred = DetectSendor(qName='pred_queue')

if __name__ == "__main__":
    main()
    
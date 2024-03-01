
video_list = [
                # "./sources/ntu_sample.avi",  # 2
                # "./sources/ntu_sample.avi",  # 2
                # "./sources/ntu_sample.avi",  # 2
                # "./sources/ntu_sample.avi",  # 2
                # "./sources/ntu_sample.avi", # 2
                # "./sources/0x6444E8DE_20230423161422_20230423171421.avi", # no
                "./sources/0x64489710_20230426120336_20230426121423.avi", # lot

                "./sources/0x64489710_20230426120336_20230426121423.avi", # lot
                "./sources/0x6447B5E7_20230425191343_20230425200109.avi", # lot
                # "./sources/0x64489710_20230426120336_20230426121423.avi", # lot
                # "./sources/0x6447B5E7_20230425191343_20230425200109.avi", # lot
                # "./sources/0x64489710_20230426120336_20230426121423.avi", # lot
                # "./sources/0x6447B5E7_20230425191343_20230425200109.avi", # lot
                # "./sources/0x64489710_20230426120336_20230426121423.avi", # lot
                # "./sources/0x6447B5E7_20230425191343_20230425200109.avi", # lot
                "./sources/0x6446FF40_20230425070514_20230425071423.avi", # normal
                "./sources/0x6446FF40_20230425070514_20230425071423.avi", # normal

                # "./sources/0x6444E8DE_20230423161422_20230423171421.avi",# no
                # "./sources/0x6447B5E7_20230425191343_20230425200109.avi", # lot
            ]


device_use_Pose = 0
device_use_motion = 1

providers = [
    ('CUDAExecutionProvider', {'device_id': device_use_Pose,}),
    #  'CPUExecutionProvider',
]

kptSeqNum = 40
motionInferenceInterval = 0.05
cycle = True # video mode cycle
cuda_cv = True
symbo = "*"
from Object_PoseEstimate_onnx import PoseEstimateONNX,output_to_json_onnx

from modules import poseTracking_module
from modules import VideoCaptureHard as cap

import time
import torch
from torchvision.transforms import Compose

import cv2
import numpy as np

from pyskl.apis import inference_recognizer, init_recognizer
import mmcv

import queue
import threading

# pose estimate
batch = len(video_list)
modelPath = f"./weights/yolov5_onnx_models/Yolov5s6_pose_640_ti_litebatch{batch}.onnx"
PoseEstimator = PoseEstimateONNX(providers=providers,
                                 modelPath = modelPath,
                                 batchSize = batch,
)

# pose tracking
objectTrackDicts = []
for camIndex in range(len(video_list)):
    objectTrackDicts.append(poseTracking_module.Sort(height_max = 576,
                                                    width_max = 960,
                                                    categories = [],
                                                    record_absence = False,
                                                    output_absence = False,
                                                    min_hits=2,max_miss=5,
                                                    max_history = 100,
                                                    camIndex = camIndex
                                                    ))


# motion classification
stgcnpp_checkpoint = "./weights/stgcnpp_123_20230601.pth"
stgcnpp_config = "./configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"
config = mmcv.Config.fromfile(stgcnpp_config)
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
stgcnpp_model = init_recognizer(config, stgcnpp_checkpoint, f"cuda:{device_use_motion}").half()
stgcnpp_model.eval()  # 切换到推理模式
previous_t3 = time.time()




class CustomTransform(torch.nn.Module):
    def __init__(self):
        super(CustomTransform, self).__init__()
        self.num_clips = 10
        self.clip_len = 100
        self.threshold = 0.01
        self.w = 960
        self.h = 576
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
        for clip_idx in range(self.num_clips):
            start = torch.randint(0, num_frames, (1,))
            inds = torch.arange(int(start), int(start) + self.clip_len)
            allinds.append(inds)
            # print("inds.shpae",inds.shape)
        inds = torch.cat(allinds)
        # print("inds.shpaessssss",inds.shape)

        inds = inds % num_frames
        transitional = torch.zeros(num_frames, dtype=torch.bool)
        inds_int = inds.long()
        coeff = transitional[inds_int]
        coeff = coeff.int()
        inds = (coeff * inds_int + (1 - coeff) * inds).float()

        keypoint = keypoint[:, inds.long()].float()

        pad_dim = self.num_person - keypoint.shape[0]
        pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype) # device = f"cuda:{device_use_motion}"
        # pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype,device = f"cuda:{device_use_motion}")

        keypoint = torch.cat((keypoint, pad), dim=0)
        keypoint = keypoint.view(self.M, self.nc, self.seqNum, self.V, self.C)
        keypoint = keypoint.transpose(1, 0)
        keypoint = keypoint.contiguous()
        keypoint = torch.unsqueeze(keypoint, 0)
        return keypoint
transform = CustomTransform()

for i in range(10):
    t1_warm = time.time()
    with torch.no_grad():
        transfered_set = torch.empty((10,10,2,100,17,3), dtype=torch.float32).half().to(f"cuda:{device_use_motion}")
        result = stgcnpp_model(keypoint = transfered_set)
    t2_warm = time.time()
    print(round(t2_warm-t1_warm,4))
print(f"Finished stgcn warm-up ......")
def main():
    global imgTensors
    global poseTrackResults
    global GFMotion ; global ThMotion
    lastInferenceMotion = time.time()

    motionQ = queue.Queue()
    tM = threading.Thread(target = process_content_Motion, args = (motionQ,))
    tM.setDaemon(True)
    tM.start()

    sourcesNum = len(video_list)
    vidcaps = multi_video_load(video_list)
    # imgTensors = torch.empty((sourcesNum, 3, 576, 960), dtype=torch.float32)#.to(f"cuda:{device_use_Pose}").half()
    imgTensors = np.zeros((batch, 3, 640, 640)).astype(np.float32)
    imgs = [np.zeros((2160,3840,3)).astype('uint8')] * sourcesNum
    poseTrackResults = [None]*sourcesNum
    ThMotion = False
    GFMotion = False
    countFrame = 0
    while True:
        t0_pose = time.time()
        imgs = sources_read(vidcaps,video_list,imgs)
        t1_pose = time.time()

        for lineIndex,img in enumerate(imgs):
            # posePreprocessV1(lineIndex,img)
            imgTensors[lineIndex] = posePreprocessONNX(img)

        t2_pose = time.time()
        print(imgTensors.shape)
        outputs = PoseEstimator.posePred(imgs = imgTensors)
        
        t3_pose = time.time()

                
        for lineIndex,output in enumerate(outputs):
            print(lineIndex,output.shape)
            output = output_to_json_onnx(output,detConf=0.2)
            poseTrackResult = objectTrackDicts[lineIndex].tracking(output)

            exists_id = []
            for trackIndex, track in enumerate(poseTrackResult):
                kpt = [] ; kptScore = []
                for trajectIndex in range(len(track["trajectories_opt"])):
                    # print(track["trajectories_opt"][trajectIndex].keys())
                    try:
                        kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
                        # kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
                    except:
                        # pass
                        kpt.append(np.zeros((17,3)))
                        # kpt.append(np.zeros((17,2)))
                        # kptScore.append(np.zeros(17))
                poseTrackResult[trackIndex]["keypoints_seq"] = np.array(kpt)
                # poseTrackResult[trackIndex]["keypointScore_seq"] = np.array(kptScore)

            poseTrackResults[lineIndex] = poseTrackResult
            # print(f"append : {lineIndex}")
        t4_pose = time.time()


        # print(f"input : {imgTensors.shape} output : {outputs.shape}")
        print(f'''
                        POSE total  cost : {round(t4_pose-t1_pose,4)} sec
                            capture cost : {round(t1_pose-t0_pose,4)} sec
                        preprocess  cost : {round(t2_pose-t1_pose,4)} sec
                        inference   cost : {round(t3_pose-t2_pose,4)} sec
                        postprocess cost : {round(t4_pose-t3_pose,4)} sec
                ''')

        if ThMotion == False and poseTrackResults != [None]*sourcesNum and countFrame>= kptSeqNum and time.time()-lastInferenceMotion > motionInferenceInterval:
            motionQ.put({"cmd"           : "invoke",
                         "poseTrackResults" :  poseTrackResults})
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
            vidcaps.append(cap.VideoCapture(source,3840,2160,0))

        else:
            vidcaps.append(cv2.VideoCapture(source))
    return vidcaps

def sources_read(vidcaps,sourcesList,imgs):
    for video_index,source in enumerate(sourcesList):
        ret, image = vidcaps[video_index].read()
        if not ret and cycle:
            vidcaps[camIndex].release()
            time.sleep(0.001)
            if "rtsp" in video_list[camIndex]:
                # vidcaps[camIndex] = cap.VideoCapture(video_list[camIndex])
                vidcaps[camIndex] = cap.VideoCapture(video_list[camIndex],3840,2160,0)

            else:
                vidcaps[camIndex] = cv2.VideoCapture(video_list[camIndex])
            continue
        elif not ret and not cycle:
            print("video Finished ...............")
        else:

            imgs[video_index] = image
    return imgs

def posePreprocessONNX(img):
    # img = cv2.resize(img0, (640,640), interpolation=cv2.INTER_LINEAR)
    img = Resize(img,(640,640),False)
    img = (img - 127.5)/127.5 
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img

def Resize(img_cpu,sizeNew,cuda_cv = False):
    if cuda_cv:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(img_cpu)
            gpu_frame = cv2.cuda.resize(gpu_frame,sizeNew)
            img_cpu = gpu_frame.download()
        except:
            # img_cpu = cv2.resize(img_cpu,sizeNew)
            img_cpu = cv2.resize(img_cpu,sizeNew, interpolation = cv2.INTER_LINEAR)

    else:
        img_cpu = cv2.resize(img_cpu,sizeNew)
    return img_cpu


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
            if cmd == 'quit':
                print("收到 quit 訊息，即將結束")
                break      
            if cmd == 'invoke':
                eachkeypoints_seqs = []
                belongLine = []
                t1_motion = time.time()                    
                tpre2 = 0
                for lineIndex in range(len(poseTrackResults)):
                    poseTrackResult = poseTrackResults[lineIndex]
                    # print("poseTrackResult",poseTrackResult)
                    trackPersonNum = len(poseTrackResult)
                    # print(f"lineIndex : {lineIndex} personNum : {trackPersonNum}")
                    if trackPersonNum:
                        for personIndex in range(trackPersonNum):
                            if "objectID" in poseTrackResult[personIndex]:
                                eachTrackID = poseTrackResult[personIndex]["objectID"]
                                eachkeypoints_seq = poseTrackResult[personIndex]["keypoints_seq"]
                                if len(eachkeypoints_seq) < kptSeqNum or np.array_equiv(eachkeypoints_seq[-1],np.zeros((1, 17, 3))):
                                    continue
                                eachkeypoints_seq = eachkeypoints_seq[-kptSeqNum:]
                                if np.array_equiv(eachkeypoints_seq[0],np.zeros((1, 17, 3))):
                                    continue
                                x,y = poseTrackResult[personIndex]["objectPicX"],poseTrackResult[personIndex]["objectPicY"]
                                w,h = poseTrackResult[personIndex]["objectWidth"],poseTrackResult[personIndex]["objectHeight"]
                                keepPart = ~np.all(eachkeypoints_seq == np.zeros((1, 17, 3)), axis=(1, 2))
                                eachkeypoints_seq = eachkeypoints_seq[keepPart]
                                tpre111 = time.time()
                                eachkeypoints_seq = torch.tensor(eachkeypoints_seq).float()#.to(f'cuda:{device_use_motion}').float()
                                # tpre_toTensor = time.time()
                                # print("tpre_toTensor",tpre_toTensor-tpre111)
                                # eachkeypoints_seq = eachkeypoints_seq.to(f'cuda:{device_use_motion}').float()
                                # tpre_todevice = time.time()
                                # print("tpre_todevice",tpre_todevice-tpre_toTensor)

                                eachkeypoints_seq = torch.unsqueeze(eachkeypoints_seq, 0)
                                eachkeypoints_seq = transform(eachkeypoints_seq)
                                eachkeypoints_seqs.append(eachkeypoints_seq)
                                belongLine.append(lineIndex)
                                tpre222 = time.time()
                                tpre2+=tpre222-tpre111

                if len(eachkeypoints_seqs):
                    eachkeypoints_seqs = torch.cat(eachkeypoints_seqs, dim=0)#.to(f'cuda:{device_use_motion}')
                    # print(eachkeypoints_seqs.shape, belongLine)
                    t2_motion = time.time()
                    with torch.no_grad():
                        result = stgcnpp_model(keypoint = eachkeypoints_seqs.to(f'cuda:{device_use_motion}').half())#.to(f'cuda:{device_use_motion}')
                        # result = stgcnpp_model(keypoint = eachkeypoints_seqs.half())#.to(f'cuda:{device_use_motion}')
                    print(result.shape)
                    t3_motion = time.time() 
                    # print(t3_motion-previous_t3,"FUCKkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
   
                    print(f'''\n{symbo*80}
                        Motion inference :
                                * Total      cost : {round(t3_motion-t1_motion,4)} sec 
                                * preprocess cost : {round(t2_motion-t1_motion,4)} sec 
                                *        tfm cost : {round(tpre2,4)} sec 
                                * inference  cost : {round(t3_motion-t2_motion,4)} sec 
                                * Total people Num : {len(result)}
                                * data distribute : {belongLine}
                                ''')
                    previous_t3 = time.time()                
                else:
                    time.sleep(0.005)    

                GFMotion = True
                ThMotion = True


if __name__ == "__main__":
    main()
    
TieCode = True
writeImgResize = True
write2Smb = True
shutthefuckup = False#True
sendEventServer = True
cycle = True # video mode cycle
# cuda_cv = True
use_gst = False
symbo = "*"

# -------------------------------------------------------------------------------
## [Report]

PUBLIC_THRESHOLD = 80
write2SmbInterval = 0.5 # second

kptSeqNum = 60
max_miss=5
motionInferenceInterval = 0.5
pose_shown_interval = 5
# -------------------------------------------------------------------------------
## [sources]

# video_list = ['rtsp://192.168.8.169:7070/stream2']*1#+['rtsp://192.168.8.166:7070/stream2']*1
video_list = ['rtsp://admin:admin@10.1.1.202:554/profile1']
# -------------------------------------------------------------------------------
## [motion parameter]
from collections import defaultdict


singleDangerous_cls = defaultdict(lambda: [])
singleDangerous_cls.update({0: 7,
                            1: 8,
                            2: 126,
                            3: 125,
                            4: 124,
                            7: 122,
                            11: 128,})


interactiveDangerous_cls = defaultdict(lambda: [])
interactiveDangerous_cls.update({10: 64})


interactiveDangerous_cls_transfer = defaultdict(lambda: [])
interactiveDangerous_cls_transfer.update({64: 129})

# -------------------------------------------------------------------------------
## [motion information]
'''
0 : sit down ; 1 : stand up ; 2 : reach into pocket ; 3 : rub hands
4 : staggering ; 5 : sitting ; 6 : standing ; 7 : lie down ; 8 : cross foot to straight
9 : walking  ; 10: arm swimming # add 20231024 
11: standup Pred # add 20231102
'''

cls_det = [56,57]

actionMatch2ORGcls = defaultdict(lambda: [])
actionMatch2ORGcls.update({0: 7, 1: 8, 
                           2: 126, 3: 125,
                           4: 124, 5: 120,
                           6: 121, 7: 122,
                           10: 64, 9: 127,
                           11: 128})

# -------------------------------------------------------------------------------
## [sacle]
model_trained_scale_w = 1920
model_trained_scale_h = 1080 # 1280
video_w = 1280
video_h = 720
eventServer_expect_w = 3840
eventServer_expect_h = 2160
write_w = 640
write_h = 360
# -------------------------------------------------------------------------------
## [device]
device_use_Pose = 0
device_use_motion = 1

# --------------------------------------------------------------------------------
## [threshold]
foot_visibility_thres = 0.2
threshold_Neighbor = 0 # pixel

########################################################################################################################
# [model & checkpoint]

# -------------------------------------------------------------------------------
# motion classification
# stgcnpp_checkpoint = "./weights/yolov8_stgcnpp_123_20230901.pth" # train with yolov8pose
# stgcnpp_config = "./configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"

# -------------------------------------------------------------------------------
# stgcnpp_checkpoint = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_10cls_for_show/j/best_top1_acc_epoch_13.pth" # train with yolov8pose
# stgcnpp_config = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_10cls_for_show/j/j.py"
# clss_path = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_10cls_for_show/j/cls.txt"

# -------------------------------------------------------------------------------
# stgcnpp_checkpoint = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_11cls_for_show_20231002/j/yolov8_stgcnpp_SHOWUSE_cls_11_20231002.pth" # train with yolov8pose
# stgcnpp_config = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_11cls_for_show_20231002/j/j.py"
# clss_path = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_11cls_for_show_20231002/j/cls.txt"

# -------------------------------------------------------------------------------
stgcnpp_checkpoint = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_12cls_for_show/j/yolov8_stgcnpp_SHOWUSE_cls_12_20231102.pth" # train with yolov8pose
stgcnpp_config = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_12cls_for_show/j/j.py"
clss_path = "./configs/stgcn++/stgcn++_ITRI_YOLOv8_12cls_for_show/j/cls.txt"

cls_det = [56,57]

engine_file_path = "weights/yolov8l-pose.pt"
aid_det = True

cap = None
if use_gst:
    from modules import VideoCaptureHard as cap
else:
    from modules import VideoCapture as cap

from collections import defaultdict
import numpy as np


singleDangerous_cls_keys = np.array(list(singleDangerous_cls.keys()))
interactiveDangerous_cls_keys = np.array(list(interactiveDangerous_cls.keys()))

singleDangerous_cls_values = np.array(list(singleDangerous_cls.values()))
interactiveDangerous_cls_values = np.array(list(interactiveDangerous_cls.values()))
interactiveDangerous_cls_transfer_key = np.array(list(interactiveDangerous_cls_transfer.keys()))
device_use_motion = 0
device_preprocess = 0
from pyskl.apis import inference_recognizer, init_recognizer
import mmcv
import torch
from torchvision.transforms import Compose

import numpy as np
import time
import matplotlib.pyplot as plt

# device = torch.device(f"cuda:{device_use_motion}" if torch.cuda.is_available() else "cpu")
stgcnpp_checkpoint = "./weights/stgcnpp_123_20230601.pth"
stgcnpp_config = "./configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"
config = mmcv.Config.fromfile(stgcnpp_config)
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
stgcnpp_model = init_recognizer(config, stgcnpp_checkpoint, f"cuda:{device_use_motion}").half()

import json
f = open("./sources/test_kpt.txt")
actIndexs,actScores,kpts,kptScores = [],[],[],[]
for line in f.readlines():
    dirr = json.loads(line)
    actIndex,actScore,kpt,kptScore = dirr['actIndex'],dirr['actScore'],dirr['kpt'],dirr['kptScore']
    actIndexs.append(actIndex)
    actScores.append(actScore)
    kpts.append(kpt)
    kptScores.append(kptScore)
f.close()

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

    def forward(self, keypoints):
        result = torch.empty((len(keypoints), 10, 2, 100, 17, 3),device = f"cuda:{device_preprocess}")
        # print(len(keypoints))
        for kptIndex,keypoint in enumerate(keypoints):
            # print("kptIndex",kptIndex)
            keypoint = keypoint.unsqueeze(0)
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
            inds = torch.cat(allinds)

            inds = inds % num_frames
            transitional = torch.zeros(num_frames, dtype=torch.bool)
            inds_int = inds.long()
            coeff = transitional[inds_int]
            coeff = coeff.int()
            inds = (coeff * inds_int + (1 - coeff) * inds).float()

            keypoint = keypoint[:, inds.long()].float()

            pad_dim = self.num_person - keypoint.shape[0]
            # print("pad_dim",pad_dim)
            pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype,device = f"cuda:{device_preprocess}")
            # pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype,device = f"cuda:{device_use_motion}")
            # print(keypoint.shape)

            keypoint = torch.cat((keypoint, pad), dim=0)
            keypoint = keypoint.view(self.M, self.nc, self.seqNum, self.V, self.C)
            keypoint = keypoint.transpose(1, 0)
            keypoint = keypoint.contiguous()
            # keypoint = torch.unsqueeze(keypoint, 0)
            result[kptIndex] = keypoint
        return result
transform = CustomTransform()



################ 3




while True:
    eachkeypoints_seqs = []
    t1 = time.time()
    for i in range(10):
        eachkeypoints_seq = np.random.rand(40,17,2)
        eachkeypointScore_seq = np.random.rand(40,17)

        keepPart = ~np.all(eachkeypoints_seq == np.zeros((1, 17, 2)), axis=(1, 2))
        eachkeypoints_seq = eachkeypoints_seq[keepPart]
        eachkeypointScore_seq = eachkeypointScore_seq[keepPart]

        tpre111 = time.time()
        eachkeypoints_seq = torch.tensor(eachkeypoints_seq).float()#.to(f'cuda:{device_preprocess}')
        eachkeypointScore_seq = torch.tensor(eachkeypointScore_seq).float()# .to(f'cuda:{device_preprocess}')
        eachkeypoints_seq = torch.unsqueeze(eachkeypoints_seq, 0)
        eachkeypointScore_seq = torch.unsqueeze(eachkeypointScore_seq, 0)
        eachkeypointScore_seq = torch.unsqueeze(eachkeypointScore_seq, -1)
        eachkeypoints_seq = torch.cat([eachkeypoints_seq, eachkeypointScore_seq], dim=-1)
        eachkeypoints_seqs.append(eachkeypoints_seq)
    t2 = time.time()
    eachkeypoints_seqs = torch.cat(eachkeypoints_seqs, dim=0).to(f'cuda:{device_preprocess}')
    print("before : ",eachkeypoints_seqs.shape)
    tk = time.time()

    eachkeypoints_seqs = transform(eachkeypoints_seqs)
    print("after : ",eachkeypoints_seqs.shape)
    t3 = time.time()
    print("tk-t2,t3-tk",tk-t2,t3-tk)

    with torch.no_grad():
        result = stgcnpp_model(keypoint = eachkeypoints_seqs.half().to(f"cuda:{device_use_motion}"))

    t4 = time.time()
    print(f"t2-t1 : {t2-t1}")
    print(f"t3-t2 : {t3-t2}")
    print(f"t4-t3 : {t4-t3}")
    print(f"t4-t1 : {t4-t1}\n")




#####################################################################
'''
Inference
'''
# transfer_set = torch.empty((20,40,17,3), dtype=torch.float32)
# transfer_set = []
# min_len = 40
################# 1

# diffDataNumTest = []
# t1 = t2 = 0
# for k in range(40):
    
#     dataNum = k+1
#     for j in range(3):

#         with torch.no_grad():
#             transfered_set = torch.empty((dataNum,10,2,100,17,3), dtype=torch.float32)
#             t1 = time.time()

#             for i in range(dataNum):

#                 kpt = kpts[i]
#                 kptScore = kptScores[i]
#                 kpt = torch.tensor(kpt).to(f"cuda:{device_use_motion}").float()
#                 kptScore = torch.tensor(kptScore).to(f"cuda:{device_use_motion}").float()
#                 kpt = torch.unsqueeze(kpt, 0)
#                 kptScore = torch.unsqueeze(kptScore, 0)
#                 kptScore = torch.unsqueeze(kptScore, -1)
#                 kpt = torch.cat([kpt, kptScore], dim=-1)
#                 t_tsf1 = time.time()
#                 transfered_set[i] = transform(kpt)
#                 t_tsf2 = time.time()

#             t2 = time.time()
#         with torch.no_grad():
#             result = stgcnpp_model(keypoint = transfered_set.to(device).half())
#         t3 = time.time()
#         time.sleep(0.7)
#     print(transfered_set.shape[0],f"total {round(t3-t1,4)}  preprocess {round(t2-t1,4)}  inference {round(t3-t2,4)}")

#     diffDataNumTest.append([dataNum,t3-t2,t2-t1,t3-t1])


################ 2 

# t1_warm = time.time()
# with torch.no_grad():
#     transfered_set = torch.empty((10,10,2,100,17,3), dtype=torch.float32).to(f"cuda:{device_use_motion}").half()
#     result = stgcnpp_model(keypoint = transfered_set)
# t2_warm = time.time()
# print(round(t2_warm-t1_warm,4))



# while True:
#     eachkeypoints_seqs = []
#     t1 = time.time()
#     for i in range(40):
#         eachkeypoints_seq = np.random.rand(40,17,2)
#         eachkeypointScore_seq = np.random.rand(40,17)

#         keepPart = ~np.all(eachkeypoints_seq == np.zeros((1, 17, 2)), axis=(1, 2))
#         eachkeypoints_seq = eachkeypoints_seq[keepPart]
#         eachkeypointScore_seq = eachkeypointScore_seq[keepPart]

#         tpre111 = time.time()
#         eachkeypoints_seq = torch.tensor(eachkeypoints_seq).to(f'cuda:{device_use_motion}').float()
#         eachkeypointScore_seq = torch.tensor(eachkeypointScore_seq).to(f'cuda:{device_use_motion}').float()
#         eachkeypoints_seq = torch.unsqueeze(eachkeypoints_seq, 0)
#         eachkeypointScore_seq = torch.unsqueeze(eachkeypointScore_seq, 0)
#         eachkeypointScore_seq = torch.unsqueeze(eachkeypointScore_seq, -1)
#         eachkeypoints_seq = torch.cat([eachkeypoints_seq, eachkeypointScore_seq], dim=-1)
        
        
#         eachkeypoints_seq = transform(eachkeypoints_seq)
#         eachkeypoints_seqs.append(eachkeypoints_seq)
#     t2 = time.time()

#     if len(eachkeypoints_seqs):
#         eachkeypoints_seqs = torch.cat(eachkeypoints_seqs, dim=0).to(f'cuda:{device_use_motion}').half()

#     t3 = time.time()

#     with torch.no_grad():
#         result = stgcnpp_model(keypoint = eachkeypoints_seqs)

#     t4 = time.time()
#     print(f"t2-t1 : {t2-t1}")
#     print(f"t3-t2 : {t3-t2}")
#     print(f"t4-t3 : {t4-t3}")
#     print(f"t4-t1 : {t4-t1}\n")






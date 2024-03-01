'''
Execute in main ==> python pyskl/stgcnpp_pth2onnx.py
'''

# device = 'cpu'
device = "cuda:0"

from pyskl.apis import inference_recognizer, init_recognizer
import mmcv
import torch
import torchvision
import numpy as np
import time
import matplotlib.pyplot as plt
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import onnxsim
import onnx
stgcnpp_checkpoint = "./weights/stgcnpp_123_20230601.pth"
stgcnpp_config = "./configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"
config = mmcv.Config.fromfile(stgcnpp_config)
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
stgcnpp_model = init_recognizer(config, stgcnpp_checkpoint, device)

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
    def __init__(self,device):
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
        self.device = device

    def forward(self, keypoints):
        keypoints = keypoints.to(self.device)
        result = torch.empty((len(keypoints), 10, 2, 100, 17, 3),device=keypoints.get_device())
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
                start = torch.randint(0, num_frames, (1,))# ,device = keypoint.get_device()
                inds = torch.arange(int(start), int(start) + self.clip_len)#,device = keypoint.get_device()
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
            pad = torch.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype,device = keypoint.get_device())

            keypoint = torch.cat((keypoint, pad), dim=0)
            keypoint = keypoint.view(self.M, self.nc, self.seqNum, self.V, self.C)
            keypoint = keypoint.transpose(1, 0)
            keypoint = keypoint.contiguous()
            # keypoint = torch.unsqueeze(keypoint, 0)
            result[kptIndex] = keypoint
        return result
transform = CustomTransform(device = device)




################ 3



count=0
while True:
    eachkeypoints_seqs = []
    for i in range(10):
        eachkeypoints_seq = np.random.rand(40,17,3)
        keepPart = ~np.all(eachkeypoints_seq == np.zeros((1, 17, 3)), axis=(1, 2))
        eachkeypoints_seq = eachkeypoints_seq[keepPart]
        eachkeypoints_seq = torch.tensor(eachkeypoints_seq).float()
        eachkeypoints_seq = torch.unsqueeze(eachkeypoints_seq, 0)
        eachkeypoints_seq = transform(eachkeypoints_seq)
        eachkeypoints_seqs.append(eachkeypoints_seq)

    if len(eachkeypoints_seqs):
        eachkeypoints_seqs = torch.cat(eachkeypoints_seqs, dim=0)#.to(f'cuda:{device_use_motion}')
    # print(eachkeypoints_seqs.shape, belongLine)
    eachkeypoints_seqs = eachkeypoints_seqs.to(device)
    with torch.no_grad():
        result = stgcnpp_model(keypoint = eachkeypoints_seqs)
    print(result.shape)
    if count>=10:
        break
    count+=1

# data = torch.rand(1, 10, 2, 100, 17, 3).to(device)
# result = stgcnpp_model(keypoint = eachkeypoints_seqs)
onnx_path = "stgcnpp_123_20230601_dynamic.onnx"
onnx_pathS = "stgcnpp_123_20230601_dynamic_simplify.onnx"
dynamic_axes = {
                'keypoint' : {0: 'batch'},
                '1416' : {0: "batch"}
               }

torch.onnx.export(stgcnpp_model,               
                  eachkeypoints_seqs,                        
                  onnx_path,  
                  verbose=False,
                  opset_version=14,        
                  input_names = ['keypoint'],   
                  output_names=["1416"],
                  dynamic_axes=dynamic_axes)

model_onnx = onnx.load(onnx_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model


model_onnx, check = onnxsim.simplify(model_onnx,
                dynamic_input_shape=True,
                # input_shapes={'images': list(img.shape)} if opt.dynamic else None
                )
onnx.save(model_onnx, onnx_pathS)
print("Simplify finish.................")


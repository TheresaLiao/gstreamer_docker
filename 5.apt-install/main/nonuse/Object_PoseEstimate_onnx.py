# posePath = "yolov7"
import sys
# sys.path.append(".")
# sys.path.append(posePath)
import numpy as np
import os
import time
import copy
import cv2
# from torchvision import transforms

import onnxruntime


def output_to_json_onnx(output, ratioH=576/640, ratioW=960/640,detConf=0.3):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
    
    kpts = np.reshape(kpts,(len(det_bboxes),17,3))
    det_bboxes[:,0],det_bboxes[:,2] = det_bboxes[:,0]*ratioW,det_bboxes[:,2]*ratioW
    det_bboxes[:,1],det_bboxes[:,3] = det_bboxes[:,1]*ratioH,det_bboxes[:,3]*ratioH
    kpts[:,:,0] = kpts[:,:,0]*ratioW
    kpts[:,:,1] = kpts[:,:,1]*ratioH
    

    # print(det_bboxes.shape,det_scores.shape,kpts.shape)
    idx = (det_scores >= 0.3).flatten()
    filtered_bboxes = det_bboxes[idx]
    filtered_kpts = kpts[idx]
    filtered_scores = det_scores[idx]
    targets = []
    for bbox, kpt, score in zip(filtered_bboxes, filtered_kpts, filtered_scores):
        x1,y1,x2,y2 = bbox
        w,h = x2-x1,y2-y1
        single_json = {"objectPicX" : x1,
                        "objectPicY" : y1,
                        "objectWidth" : w,
                        "objectHeight": h,
                        "objectTypes":"person",
                        "Confidents": score,
                        "keypoints": kpt,# keypoint
                        # "keypointScore": keypointScore,
        }
        targets.append(single_json)     
    return targets



class PoseEstimateONNX(object):
    def __init__(self,
                 half=True,
                 confidence_thres = 0.25,
                 iou_thres = 0.65,
                 batchSize = 1,
                 modelPath = f"./yolov5_onnx_models/Yolov5s6_pose_640_ti_litebatch1.onnx",
                 providers = [('CUDAExecutionProvider', {'device_id': 0,})]
                 ):
        
        self.half = half
        self.iou_thres = iou_thres
        self.confidence_thres = confidence_thres
        self.img_mean = 127.5    
        self.img_scale=1/127.5  
        self.batchSize = batchSize
        self.score_threshold = 0.3
        self.modelPath = modelPath

        # self.so = onnxruntime.SessionOptions()
        # self.so.device_id = device
        self.session = onnxruntime.InferenceSession(self.modelPath, None,
                                            providers=providers
                                                )
        self.input_name = self.session.get_inputs()[0].name


    def posePred(self,imgs):
        return self.session.run([], {self.input_name: imgs})

    # def ImgPoseprocess(self, image):
    #     image = letterbox(image, 960, stride=64, auto=True)[0]
    #     image_ = image.copy()
    #     image = image.transpose(2, 0, 1)#.copy()
    #     image = torch.from_numpy(image)#.to(self.device)
    #     image = image.float()  # uint8 to fp16/32
    #     image /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     image = image.unsqueeze(0)
    #     if torch.cuda.is_available():
    #         if self.half:
    #             image = image.half()
    #         image = image.to(self.device)
    #     return image,image_

    # def nmsFromTensor(self,predTensor):
    #     # print("start calculate nms ................")
    #     return non_max_suppression_kpt(predTensor, 
    #                                    self.confidence_thres, 
    #                                    self.iou_thres,
    #                                    nc=self.nc, 
    #                                    nkpt=self.nkpt, 
    #                                    kpt_label=True) 

    # def poseEstimation(self, image):
    #     orgH, orgW, _  = image.shape
    #     image,image_ = self.ImgPoseprocess(image)
    #     ratioH,ratioW = orgH/576, orgW/960
    #     with torch.no_grad():
    #         self.output, _ = self.pose_model(image)
    #         self.output = non_max_suppression_kpt(self.output, 
    #                                               self.confidence_thres, 
    #                                               self.iou_thres,
    #                                               nc=self.nc, 
    #                                               nkpt=self.nkpt, 
    #                                               kpt_label=True)
    #         # self.output = output_to_keypoint(self.output) # --> np.array (n,51)
    #         self.output = output_to_json(self.output, ratioH, ratioW)  # --> json form
    #     return self.output,image_

    # def poseTracking(self, det_json):
    #     tracking_result = self.objectTrackDict['tracking'].tracking(det_json)
        
    #     exists_id = []
    #     for trackIndex, track in enumerate(tracking_result):
    #         kpt = []
    #         kptScore = []
    #         for trajectIndex in range(len(track["trajectories_opt"])):
    #             # print(track["trajectories_opt"][trajectIndex].keys())
    #             try:
    #                 kpt.append(track["trajectories_opt"][trajectIndex]["keypoints"])
    #                 kptScore.append(track["trajectories_opt"][trajectIndex]["keypointScore"])
    #             except:
    #                 kpt.append(np.zeros((17,2)))
    #                 kptScore.append(np.zeros(17))
    #         kpt = np.array(kpt)
    #         kptScore = np.array(kptScore)
    #         tracking_result[trackIndex]["keypoints_seq"] = kpt
    #         tracking_result[trackIndex]["keypointScore_seq"] = kptScore
    #     return tracking_result

    # def plot_pose(self,output,image=None):
    #     self.nimg = None
    #     if image is None:
    #         self.nimg = np.zeros((676,960,3)).astype("uint8")
    #     else:
    #         self.nimg = image
    #         # if image.shape != (576,960,3):
    #         #     self.nimg = cv2.resize(image,(576,960))
    #         # self.nimg = image[0].permute(1, 2, 0) * 255
    #         # self.nimg = self.nimg.cpu().numpy().astype(np.uint8)
    #         # self.nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    #     for idx in range(output.shape[0]):
    #         self.nimg = plot_skeleton_kpts(self.nimg, output[idx, 7:].T, 3)
    #     return self.nimg

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return im
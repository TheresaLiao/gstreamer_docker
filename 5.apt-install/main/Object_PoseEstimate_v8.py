posePath = "ultralytics"

# import sys
# sys.path.append(".")
# sys.path.append(posePath)
from ultralytics import YOLO
import cv2
import time
from cfg_testInteractive import engine_file_path

print("Pose model use : ",engine_file_path)
class PoseEstimate(object):
    def __init__(self, 
                 engine_file_path=engine_file_path,
                 device="cuda:0",
                 ):
        self.device = device
        self.model = YOLO(engine_file_path)

    def posePred(self, imgs):
        return self.model.predict(imgs, stream=True,device=self.device,verbose=False)  # return a generator of Results objects

    def poseTrack(self, imgs):
        return self.model.track(imgs,
                                # stream=False,
                                # device=self.device,
                                tracker="bytetrack.yaml",
                                persist=True,
                                verbose=False) 

class ObjDetect(object):
    def __init__(self, 
                 engine_file_path="weights/yolov8l.pt",
                 device="cuda:0",
                 ):
        self.device = device
        self.model = YOLO(engine_file_path)

    def detPred(self, imgs, classes=None,conf=0.3,half=False):
        return self.model.predict(imgs, 
                                  stream=False,
                                  # device=self.device,
                                  verbose=False,
                                  classes = classes,
                                  conf = conf,
                                  half = half
                                  ) 


def output_to_json_V8(output, ratioH=576, ratioW=960):
    boxes = output.boxes.xyxyn.cpu().numpy()
    keypoints = output.keypoints.data.cpu().numpy()
    targets = []        
    for bbox,keypoint in zip(boxes,keypoints):
        # keypoint = keypoint[0]
        # print(keypoint.shape)
        conf = 0.99
        keypoint[:,0] *= ratioW     # yolov7pose-pt- (960,576)
        keypoint[:,1] *= ratioH     # yolov7pose-pt- (960,576)
        single_json = {"objectPicX" : bbox[0],
                        "objectPicY" : bbox[1],
                        "objectWidth" : bbox[2],
                        "objectHeight": bbox[3],
                        "objectTypes":"person",
                        "Confidents": conf,
                        "keypoints": keypoint,
        }
        targets.append(single_json)
    return targets
CONF_THRESH = 0.65
IOU_THRESHOLD =  0.45
MAX_OUTPUT_BBOX_COUNT = 1000
KEY_POINTS_NUM = 17
ratioH, ratioW = 1,1


import sys
sys.path.append(".")
import numpy as np
import cv2
import os, copy
import threading
import tensorrt as trt
import pycuda.driver as cuda
import ctypes
import time
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

def posePreprocessTRT(img):
    # img = cv2.resize(img0, (832,832), interpolation=cv2.INTER_LINEAR)
    img = Resize(img,(832,832),False)
    img = img/255 
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img
# def posePreprocessTRT(img):
#     return cv2.dnn.blobFromImage(img,scalefactor=1/255,size=(832,832),swapRB=True) ## TODO : DO this with pytorch       

# def output_to_json_TRT(bboxes, kpts, det_scores, ratioH=576/832, ratioW=960/832,detConf=0.3):
#     bboxes[:,0],bboxes[:,2] = bboxes[:,0]*ratioW,bboxes[:,2]*ratioW
#     bboxes[:,1],bboxes[:,3] = bboxes[:,1]*ratioH,bboxes[:,3]*ratioH
#     kpts[:,:,0] = kpts[:,:,0]*ratioW
#     kpts[:,:,1] = kpts[:,:,1]*ratioH
    
#     idx = (det_scores >= detConf).flatten()
#     filtered_bboxes = bboxes[idx]
#     filtered_kpts = kpts[idx]
#     filtered_scores = det_scores[idx]
#     print(filtered_scores)
#     targets = []
#     for bbox, kpt, score in zip(filtered_bboxes, filtered_kpts, filtered_scores):
#         x1,y1,x2,y2 = bbox
#         w,h = x2-x1,y2-y1
#         single_json = {"objectPicX" : x1,
#                         "objectPicY" : y1,
#                         "objectWidth" : w,
#                         "objectHeight": h,
#                         "objectTypes":"person",
#                         "Confidents": score,
#                         "keypoints": kpt,# keypoint
#                         # "keypointScore": keypointScore,
#         }
#         targets.append(single_json)     
#     return targets

def output_to_json(output, ratioH, ratioW,scaleback=False):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []        
    for i in range(len(output)):
        bbox = [output[i, 0],
                output[i, 1],
                output[i, 2] - output[i, 0],
                output[i, 3] - output[i, 1]]

        conf = output[i, 4]
        keypointRaw = np.reshape(output[i, 6:],(17,3))
        keypoint = keypointRaw[:,:2]#.astype(float)
        keypoint[:,1] *= (576/960)     # yolov7pose-pt- (960,576)
        keypointScore = keypointRaw[:,-1]
        keypointORGSIZE = keypoint
        keypointORGSIZE[:,0] *= ratioW
        keypointORGSIZE[:,1] *= ratioH
        single_json = {"objectPicX" : bbox[0],
                        "objectPicY" : bbox[1],
                        "objectWidth" : bbox[2],
                        "objectHeight": bbox[3],
                        "objectTypes":"person",
                        "Confidents": conf,
                        "keypoints": keypoint,
                        "keypointScore": keypointScore,
                        "keypoints": keypointORGSIZE,
        }
        targets.append(single_json)
    return targets
    
class PoseEstimate(object):
    def __init__(self, engine_file_path,device,structure="fp16",batch=1):
        trt.init_libnvinfer_plugins(None, "")  #important !

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        if structure == "int8":
            trt.init_libnvinfer_plugins(None, "")

        runtime = trt.Runtime(TRT_LOGGER)
        self.ctx = cuda.Device(int(device)).make_context()
        stream = cuda.Stream()
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

            
        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        context.set_binding_shape(0, (batch, 3,  832, 832))
        #inputs, outputs, bindings, stream = allocate_buffers(engine,max_batch_size=BATCH_SIZE) #构建输入，输出，流指针
        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = batch

    def posePred(self, imgs):
        t1_posePred = time.time()
        threading.Thread.__init__(self)
        self.ctx.push()
        np.copyto(self.host_inputs[0], imgs.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        self.ctx.pop()

        outputs = self.host_outputs[0]
        outputs = np.reshape(outputs,(self.batch_size,-1,57))
        t2_posePred = time.time()

        outputs_NMS = []
        for batchIndex,output in enumerate(outputs):
            output = self.nms(output, 576, 960, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
            output = output_to_json(output, ratioH, ratioW,scaleback=False)
            outputs_NMS.append(output)
        # _,det_boxes,det_pose,det_scores = outputs
        # det_boxes = np.reshape(det_boxes,(self.batch_size,-1,4))
        # det_pose = np.reshape(det_pose,(self.batch_size,-1,17,3))
        # det_scores = np.reshape(det_scores,(self.batch_size,-1))
        t3_posePred = time.time()
        print(f"[PosePred] infer : {t2_posePred-t1_posePred} post : {t3_posePred-t2_posePred}")
        return outputs_NMS#det_boxes,det_pose,det_scores

    # def posePred(self, imgs):     
    #     threading.Thread.__init__(self)
    #     self.ctx.push()
    #     np.copyto(self.host_inputs[0], imgs.ravel())
    #     cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
    #     self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
    #     cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
    #     # Synchronize the stream
    #     self.stream.synchronize()
    #     self.ctx.pop()
    #     _,det_boxes,det_pose,det_scores = self.host_outputs
    #     det_boxes = np.reshape(det_boxes,(self.batch_size,-1,4))
    #     det_pose = np.reshape(det_pose,(self.batch_size,-1,17,3))
    #     det_scores = np.reshape(det_scores,(self.batch_size,-1))
    #     return det_boxes,det_pose,det_scores


    def nms(self, prediction, origin_h, origin_w, conf_thres=0.75, nms_thres=0.65):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.cxcywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []

        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, 5] == boxes[:, 5]
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
    def cxcywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [cx, cy, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        """
        y = np.copy(x)
        y[:, 0] = (x[:, 0] - (x[:, 2] / 2)) * (origin_w / 832)
        y[:, 2] = (x[:, 0] + (x[:, 2] / 2)) * (origin_w / 832)
        y[:, 1] = (x[:, 1] - (x[:, 3] / 2)) * (origin_h / 832)
        y[:, 3] = (x[:, 1] + (x[:, 3] / 2)) * (origin_h / 832)
        return y

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou


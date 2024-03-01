from ultralytics import YOLO
import cv2
import time
# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model


img = cv2.imread("zidane.jpg")

for i in range(100):
    t1 = time.time()
    # Run batched inference on a list of images
    results = model([img,img,img,img,img,img,img,img,img,img,img,img], stream=True)  # return a generator of Results objects

    # Process results generator
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
    t2 = time.time()
    print(t2-t1)
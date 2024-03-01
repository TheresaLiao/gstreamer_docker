import cv2
from modules import VideoCaptureHard as cap
import json
import numpy as np

source = 'rtsp://192.168.8.169:7070/stream2'
# source = 'rtsp://192.168.8.166:7070/stream2'

vidcap = cap.VideoCapture(source,0,0,0)
ROI_info_savePath = "ROI_setup/ROI_show_use.txt"

roi_points = []
def draw_roi_points(img, points):
    """ 在給定的圖像上繪製ROI點和多邊形。"""
    temp_img = img.copy()
    for point in points:
        cv2.circle(temp_img, tuple(point), 5, (0, 0, 255), -1)
    if len(points) > 1:
        cv2.polylines(temp_img, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
    cv2.imshow(window_name, temp_img)

def mouse_callback(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        draw_roi_points(img, roi_points)


img_draw = None
while True:
    ret, image = vidcap.read()
    if not ret:
        break
    if ret:
        img_draw = image
        break
if img_draw is None:
    print("Capture image fail")
    exit()
# img_draw = cv2.flip(img_draw,0)
img = img_draw.copy()
h,w,_ =img.shape
window_name = "Draw ROI points"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)
cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Defined ROI:", roi_points)

txtdata = {
    "RTSP" : source,
    "poly" : roi_points,
    "scale_w" : w,
    "scale_h": h,
}
with open(ROI_info_savePath,"a",encoding="utf-8") as f:
        f.write(json.dumps(txtdata)+'\n')


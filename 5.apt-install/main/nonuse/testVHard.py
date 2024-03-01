
import cv2,time
from modules import VideoCaptureHard as cap

vcap = cap.VideoCapture("rtsp://10.1.1.32:7070/stream1",3840,2160,0)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("test_ggyy.mp4", fourcc, 30, (3840, 2160))

while True:
    ret, frame = vcap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('frame',cv2.resize(frame,(960,540)))
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.01)
vcap.release()
cv2.destroyAllWindows()

import cv2,time
from modules import VideoCaptureHard as cap


# url = "'rtsp://192.168.8.169:7070/stream2'"
url = 'rtsp://admin:admin@10.1.1.202:554/profile1'
vcap = cap.VideoCapture(url,0,0,0)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("test_ggyy.mp4", fourcc, 30, (3840, 2160))

count = 0
while True:
    print(count)
    # t1 = time.time()
    ret, frame = vcap.read()
    # t2 = time.time()
    # print(t2-t1)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # cv2.imshow('frame', cv2.resize(frame,(1920,1080)))
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break
    # out.write(frame)        

    # if count >= 50:
    #     out.release()
    #     break
    time.sleep(0.01)
    # count += 1
vcap.release()
cv2.destroyAllWindows()
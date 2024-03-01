import cv2
#gurl1 = "filesrc location=/workdir/hat.mp4 ! decodebin name=dec ! videoconvert ! autovideosink dec. ! audioconvert ! audioresample ! appsink"
#gurl1 = "videotestsrc ! appsink"
#gurl1 = 'rtspsrc location=rtsp://127.0.0.1:8554/test latency=0 ! rtph264depay ! appsink'
gurl1 = 'rtspsrc location=rtsp://admin:admin@10.1.1.202:554/profile1 latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink'
cap = cv2.VideoCapture(gurl1, cv2.CAP_GSTREAMER)
print(cap.isOpened())
ret, frame = cap.read()
cv2.imwrite("test.jpg", frame)


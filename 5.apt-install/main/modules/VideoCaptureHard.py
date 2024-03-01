import cv2, queue, threading, time
import subprocess


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        #############################################
        ##### GPU #####
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   ' cudaupload ! cudaconvert ! cudadownload ! appsink').format(uri, latency)
        
        #############################################
        ##### CPU #####
        # gst_str = ('rtspsrc location={} latency={} ! '
        #            'rtph264depay ! h264parse ! avdec_h264 ! '
        #            ' videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)



# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name,  width, height, latency):
    self.cap = open_cam_rtsp(name,  width, height, latency) # , cv2.CAP_FFMPEG
    # self.cap = cv2.VideoCapture(name) 
    # self.cap.set(cv2.CAP_PROP_FPS, 25) # jinyor
    self.q = queue.Queue()
    #self.ret = False
    self.lock = threading.Lock()
    self.t = threading.Thread(target=self._reader)
    self.t.daemon = True
    self.t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      #print(cv2.CAP_PROP_BUFFERSIZE)
      # t1 = time.time()
      ret, frame = self.cap.read()
      # t2 = time.time()
      # print(t2-t1)
      if not ret:
        self.cap.release()
        break
      self.lock.acquire()
      if self.q.qsize() > 1:
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)
      self.lock.release()
      time.sleep(0.02)

  def read(self):
    if not self.q.empty():
       self.lock.acquire()
       image = self.q.get()
       ret = True
       self.lock.release()
    else:
       image = None
       ret = False
       while self.t.is_alive():
          if not self.q.empty():
             self.lock.acquire()
             image = self.q.get()
             ret = True
             self.lock.release()
             break
          else:
             time.sleep(0.001)
    return ret, image

  def release(self):
    self.lock.acquire()
    self.cap.release()
    self.lock.release()

  def get(self, param):
    return self.cap.get(param)


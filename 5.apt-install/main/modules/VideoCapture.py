import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name) # , cv2.CAP_FFMPEG
    # self.cap.set(cv2.CAP_PROP_FFMPEG_HW_ACCELERATION, 1) ## error
    print(cv2.CAP_PROP_BUFFERSIZE, cv2.CAP_PROP_FPS)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # jinyor
    # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # jinyor
    self.cap.set(cv2.CAP_PROP_FPS, 25)
    print(cv2.CAP_PROP_BUFFERSIZE, cv2.CAP_PROP_FPS)
    # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
      ret, frame = self.cap.read()
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
      time.sleep(0.002)

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


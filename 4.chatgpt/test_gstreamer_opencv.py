import cv2

def test_gstreamer_opencv():
    # 替换下面的 GStreamer 管道字符串为你的视频源
    # 这里的示例是通过 GStreamer 打开一个视频文件
    # 注意：这里的路径和管道配置仅为示例，你需要根据实际情况调整
    gst_pipeline = 'filesrc location=/app/hat.mp4 ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! appsink'

    # 使用 GStreamer 管道作为视频捕获源打开 OpenCV 视频捕获
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("视频流打开失败")
        return

    print("视频流打开成功")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法从视频流中获取帧")
            break

        # 显示视频帧
        cv2.imshow('Frame', frame)

        # 按 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_gstreamer_opencv()


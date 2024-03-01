# 基於Ubuntu22.04使 python 可以用 opencv 呼叫Gstreamer 做rtsp 的串流處理。

## 目的
* 進入點在 ./1.source/main/[testRTSP.py](https://github.com/TheresaLiao/gstreamer_docker/blob/main/1.source/main/testRTSP.py) 裡

## 問題
* 執行過程中問題大多發生在 ./1.source/main/modules/[VideoCaptureHard.py](https://github.com/TheresaLiao/gstreamer_docker/blob/main/1.source/main/modules/VideoCaptureHard.py) 這支程式碼裡，針對gst_str 對rtsp 串流的設定跑不過[Question Code](https://github.com/TheresaLiao/gstreamer_docker/blob/main/1.source/main/modules/VideoCaptureHard.py#L22C9-L22C17)

## 進度
* 撰寫了一支仿VideoCaptureHard.py 裡，針對CPU 去連接RTSP 的程式 [run.py](https://github.com/TheresaLiao/gstreamer_docker/blob/main/5.apt-install/run.py)。
* 此container環境(./5.apt-install/[Dockerfile](https://github.com/TheresaLiao/gstreamer_docker/blob/main/5.apt-install/Dockerfile))可以跑成功。

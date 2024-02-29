# gstreamer image II
[toc]
## chatgpt 給的版本
```dockerfile=
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/opencv/opencv.git /opt/opencv
RUN git clone https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib

RUN mkdir /opt/opencv/build && cd /opt/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_C_EXAMPLES=OFF \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D BUILD_EXAMPLES=OFF \
          -D WITH_GSTREAMER=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

WORKDIR /app

COPY . /app

RUN pip3 install numpy opencv-python-headless

CMD ["python3", "./your_script.py"]
```
* your_script.py
```python=
import cv2
print(cv2.getBuildInformation())
```
* Gstremer 的設定沒有打開，很奇怪
![image](https://hackmd.io/_uploads/SkfVZn6np.png)



## 單單 gstreamer 可以，但是opencv 無法使用gstreamer 呼叫
* 參考: https://blog.csdn.net/alan_yunshan/article/details/128370019
```dockerfile!
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

## setting timezone TW
RUN apt-get update \
    &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata    
RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 

## install gstreamer depend lib
RUN apt-get install -y build-essential \
    dpkg-dev flex bison autotools-dev \
    automake liborc-dev autopoint libtool \
    gtk-doc-tools python3-pip libmount-dev \
    git wget bison flex ninja-build  libglib2.0-dev nasm
RUN  apt-get install -y valgrind \
    libgirepository1.0-dev libcap-dev \
    libgtk-3-dev libunwind-dev clzip \
    gobject-introspection libdw-dev
RUN  apt-get install -y libxv-dev \
    libasound2-dev libtheora-dev \
    libogg-dev libvorbis-dev
RUN  apt-get install -y libbz2-dev \
    libv4l-dev libvpx-dev libjack-jackd2-dev \
    libsoup2.4-dev libpulse-dev
RUN  apt-get install -y faad libfaad-dev \
    libfaac-dev libx264-dev libmad0-dev

## install gstreamer
WORKDIR /tmp
RUN  git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git

## install meson
WORKDIR /tmp
RUN python3 -m pip install pip --upgrade
RUN pip3 install meson 

## install gstreamer
WORKDIR /tmp/gstreamer
RUN  meson setup --prefix /usr build -Dgst-plugins-bad:nvcodec=enabled
RUN  ninja -C build install
RUN  gst-inspect-1.0 nvcodec
## check ok! :gst-launch-1.0 -v rtspsrc location=rtsp://admin:admin@10.1.1.202:554/profile1 latency=0 ! rtph264depay ! h264parse ! avdec_h264 !  cudaupload ! cudaconvert ! cudadownload ! appsink 

## install opencv build & install
RUN  apt-get install -y build-essential cmake git python3-dev python3-numpy libavcodec-dev libavformat-dev libswscale-dev libgtk-3-dev libgtk2.0-dev libcanberra-gtk-module libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev libopencv-dev x264 libx264-dev libssl-dev ffmpeg

RUN git clone https://github.com/opencv/opencv.git /opt/opencv
RUN git clone https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib

RUN mkdir /opt/opencv/build && cd /opt/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_C_EXAMPLES=OFF \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D BUILD_EXAMPLES=OFF \
          -D WITH_GSTREAMER=ON .. 
      
## 後面這邊 opencv 編譯與安裝會死在make -j$(nproc) 那邊error 
RUN make -j$(nproc) && \
    make install && \
    ldconfig
```
![image](https://hackmd.io/_uploads/rkD1C2p2a.png)

![image](https://hackmd.io/_uploads/ryHbxTp3T.png)


## Target
* 最終要讓python 裡的opencv 可以使用gstreamer來看串流。
```python=
location=rtsp://admin:admin@10.1.1.202:554/profile1 
latency=0 
gst_str = ('rtspsrc location={} latency={} ! '
'rtph264depay ! h264parse ! avdec_h264 ! '
' cudaupload ! cudaconvert ! cudadownload ! appsink').format(uri, latency)
```

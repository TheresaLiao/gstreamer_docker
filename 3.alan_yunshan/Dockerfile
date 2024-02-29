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

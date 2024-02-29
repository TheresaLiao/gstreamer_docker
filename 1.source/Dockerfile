#FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
FROM ubuntu:20.04
RUN DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata

RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update
#RUN apt-get install -y opencv3.8
RUN apt-get install -y --no-install-recommends --fix-missing \
      autoconf \
      automake \
      build-essential \
      cmake \
      git-core \
      libass-dev \
      libfreetype6-dev \
      libgnutls28-dev \
      libmp3lame-dev \
      libsdl2-dev \
      libtool \
      libva-dev \
      libvdpau-dev \
      libvorbis-dev \
      libxcb1-dev \
      libxcb-shm0-dev \
      libxcb-xfixes0-dev \
      meson \
      ninja-build \
      pkg-config \
      texinfo \
      wget \
      yasm \
      zlib1g-dev \
      libbz2-dev \
      liblzma-dev \
      git \
      vim \
      unzip \
      nasm
RUN apt-get install -y --no-install-recommends --fix-missing \
      ninja-build \
      build-essential \
      dpkg-dev \
      flex \
      bison \
      autotools-dev \
      automake \
      liborc-dev \
      autopoint \
      libtool \
      gtk-doc-tools \
      python3-pip \
      valgrind \
      libgirepository1.0-dev \
      libcap-dev \
      libgtk-3-dev \
      libunwind-dev \
      clzip \
      gobject-introspection \
      libdw-dev \
      libxv-dev \
      libasound2-dev \
      libtheora-dev \
      libogg-dev \
      libvorbis-dev \
      libbz2-dev \
      libv4l-dev \
      libvpx-dev \
      libjack-jackd2-dev \
      libsoup2.4-dev \
      libpulse-dev \
      faad \
      libfaad-dev \
      libfaac-dev \
      libx264-dev \
      libmad0-dev \
      yasm

RUN apt-get install -y python3.8
RUN pip3 uninstall opencv-python
RUN pip install meson==0.60.0
RUN pip install requests
RUN pip install numpy==1.23.1

WORKDIR /tmp
COPY gst-plugins-good-1.19.2.tar.xz ffmpeg.tgz gst-plugins-ugly-1.19.2.tar.xz gmp-6.2.1.tar.lz gstreamer-1.19.2.tar.xz gsl-latest.tar.gz opencv.tgz gst-libav-1.19.2.tar.xz gst-plugins-bad-1.19.2.tar.xz video_codec_sdk.tgz gst-plugins-base-1.19.2.tar.xz ./

WORKDIR /tmp
RUN tar xvzf ffmpeg.tgz && cd ffmpeg/ &&  ./configure --enable-shared --prefix=/usr/local/ffmpeg && make && make install
RUN rm -rf ffmpeg.tgz

WORKDIR /tmp
RUN tar xvf gmp-6.2.1.tar.lz && cd gmp-6.2.1/ &&  ./configure && make && make install
RUN rm -rf gmp-6.2.1.tar.lz

WORKDIR /tmp
RUN tar xvf gsl-latest.tar.gz && cd gsl-2.7.1/ &&  ./configure && make && make install
RUN rm -rf gsl-latest.tar.gz

WORKDIR /tmp
RUN tar xvf gstreamer-1.19.2.tar.xz && cd gstreamer-1.19.2/ &&  meson build  && ninja -C build  && ninja -C build install
RUN rm -rf gstreamer-1.19.2.tar.xz

WORKDIR /tmp
RUN tar xvf gst-plugins-base-1.19.2.tar.xz && cd gst-plugins-base-1.19.2/ &&  meson build  && ninja -C build  && ninja -C build install
RUN rm -rf gst-plugins-base-1.19.2.tar.xz

WORKDIR /tmp
RUN tar xvf gst-plugins-good-1.19.2.tar.xz && cd gst-plugins-good-1.19.2/ &&  meson build  && ninja -C build  && ninja -C build install
RUN rm -rf gst-plugins-good-1.19.2.tar.xz

WORKDIR /tmp
RUN tar xvf gst-plugins-bad-1.19.2.tar.xz && cd gst-plugins-bad-1.19.2/ &&  meson build  && ninja -C build  && ninja -C build install
RUN rm -rf gst-plugins-bad-1.19.2.tar.xz

WORKDIR /tmp
RUN tar xvf gst-plugins-ugly-1.19.2.tar.xz && cd gst-plugins-ugly-1.19.2/ &&  meson build  && ninja -C build  && ninja -C build install
RUN rm -rf gst-plugins-ugly-1.19.2.tar.xz

WORKDIR /tmp
RUN export PKG_CONFIG_PATH=$(pwd)/ffmpeg/libavfilter:$(pwd)/ffmpeg/libavdevice:$(pwd)/ffmpeg/libswscale:$(pwd)/ffmpeg/libavutil:$(pwd)/ffmpeg/libavformat:$(pwd)/ffmpeg/libavcodec:$(pwd)/ffmpeg/libswresample
RUN echo $PKG_CONFIG_PATH

## install opencv
WORKDIR /tmp
RUN tar xvzf opencv.tgz && cd opencv/opencv-4.x/ && mkdir build

WORKDIR /tmp
COPY torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl .
RUN pip install torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl
RUN rm -rf torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl

WORKDIR /tmp
COPY torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl .
RUN pip install torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl
RUN rm -rf torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl

WORKDIR /tmp
COPY torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl .
RUN pip install torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl
RUN rm -rf torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl

RUN pip install scipy
RUN pip install fvcore
RUN pip install seaborn
RUN pip install pytorch-lightning==1.1.4
RUN pip install einops
RUN pip install joblib
RUN pip install timm
RUN pip install thop
RUN pip install yacs==0.1.8

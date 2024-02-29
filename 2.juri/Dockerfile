FROM ubuntu:20.04
#FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

RUN apt-get update 
RUN apt-get upgrade -y

## setting timezone TW
RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends tzdata
RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update

## install opencv
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
RUN pip3 install opencv-python

## install gstreamer
RUN apt-get install -y \
      libgstreamer1.0-0 \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      gstreamer1.0-libav \
      gstreamer1.0-doc \
      gstreamer1.0-tools \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev

## install torch lib
WORKDIR /tmp
COPY torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl .
COPY torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl .
COPY torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl .
RUN pip install torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl
RUN pip install torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl
RUN pip install torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl
RUN rm -rf torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl
RUN rm -rf torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl
RUN rm -rf torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl


## install mmcv
COPY mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl .
RUN pip install openmim

## install lib
RUN pip install meson==0.60.0
RUN pip install requests
RUN pip install numpy==1.23.1
RUN pip install scipy
RUN pip install fvcore
RUN pip install seaborn
RUN pip install pytorch-lightning==1.1.4
RUN pip install einops
RUN pip install joblib
RUN pip install timm
RUN pip install thop
RUN pip install yacs==0.1.8
RUN pip install setuptools==59.5.0

## install ffmpeg
RUN apt-get install -y ffmpeg

## install main lib 
COPY main /tmp/main
WORKDIR /tmp/main/ultralytics
RUN pip install -e .


ENV NVIDIA_VISIBLE_DEVICES all ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video
ENV NVIDIA_VISIBLE_DEVICES all 
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video

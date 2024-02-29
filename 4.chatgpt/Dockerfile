#FROM ubuntu:20.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

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


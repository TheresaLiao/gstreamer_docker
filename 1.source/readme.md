# gstreamer image
[TOC]

## install.xtx
```shell!
1. Download Anaconda : https://www.anaconda.com/download
    $ sh Anaconda3-2023.07-2-Linux-x86_64.sh
 
 
2. 安裝cuda
    $ sudo apt-get remove --purge '^nvidia-.*'
    $ sudo add-apt-repository ppa:graphics-drivers
    $ sudo apt-get update
    $ sudo apt-get install nvidia-driver-515
    
    $ sudo reboot
    $ nvidia-smi
    
    * Download Run file from Nvidia cuda toolkit
    $ sudo sh cuda_11.6.0_510.39.01_linux.run
    $ sudo gedit ~/.bashrc
    ==>
        export CUDA_HOME="/usr/local/cuda-11.6"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        export CPATH="$CUDA_HOME/include:$CPATH"
    $ source ~/.bashrc
    $ nvcc
    
    * Download Tar file from cudnn page (cuDNN v8.4.1 (May 27th, 2022), for CUDA 11.x)
     * unzip
     $ mv (folderName) cuda
     $ sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.6/include
     $ sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.6/lib64
     $ sudo chmod a+r /usr/local/cuda-11.6/include/cudnn*.h /usr/local/cuda-11.6/lib64/libcudnn*
    #$ cat /usr/local/cuda-11.6/include/cudnn.h | grep CUDNN_MAJOR -A 2
     $ ls /usr/local/cuda-11.8/lib64 | grep libcudnn.so




 
2. OpenCV cuda install
    * 將tar cp到~/docker_space/base
    $ conda create --name cudacv python==3.8.10
    $ sudo apt-get update
    $ sudo apt-get install -y --no-install-recommends --fix-missing \
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

4.sudo apt-get install -y --no-install-recommends --fix-missing \
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



pip3 uninstall opencv-python
pip install meson==0.60.0
pip install requests
pip install numpy==1.23.1

tar xvzf ffmpeg.tgz && cd ffmpeg/ &&  ./configure --enable-shared --prefix=/usr/local/ffmpeg && make && sudo make install
cd ..
tar xvf gmp-6.2.1.tar.lz && cd gmp-6.2.1/ &&  ./configure && make && sudo make install
cd ..
tar xvf gsl-latest.tar.gz && cd gsl-2.7.1/ &&  ./configure && make && sudo make install
cd ..
tar xvf gstreamer-1.19.2.tar.xz && cd gstreamer-1.19.2/ &&  meson build  && ninja -C build  && sudo ninja -C build install
cd ..
tar xvf gst-plugins-base-1.19.2.tar.xz && cd gst-plugins-base-1.19.2/ &&  meson build  && ninja -C build  && sudo ninja -C build install
cd ..
tar xvf gst-plugins-good-1.19.2.tar.xz && cd gst-plugins-good-1.19.2/ &&  meson build  && ninja -C build  && sudo ninja -C build install
cd ..
tar xvf gst-plugins-bad-1.19.2.tar.xz && cd gst-plugins-bad-1.19.2/ &&  meson build  && ninja -C build  && sudo ninja -C build install
cd ..
tar xvf gst-plugins-ugly-1.19.2.tar.xz && cd gst-plugins-ugly-1.19.2/ &&  meson build  && ninja -C build  && sudo ninja -C build install
cd ..
export PKG_CONFIG_PATH=$(pwd)/ffmpeg/libavfilter:$(pwd)/ffmpeg/libavdevice:$(pwd)/ffmpeg/libswscale:$(pwd)/ffmpeg/libavutil:$(pwd)/ffmpeg/libavformat:$(pwd)/ffmpeg/libavcodec:$(pwd)/ffmpeg/libswresample
echo $PKG_CONFIG_PATH
tar xvf gst-libav-1.19.2.tar.xz && cd gst-libav-1.19.2/ &&  meson build  && ninja -C build  && sudo ninja -C build install
cd ..
export PKG_CONFIG_PATH=$(pwd)/ffmpeg/libavfilter:$(pwd)/ffmpeg/libavdevice:$(pwd)/ffmpeg/libswscale:$(pwd)/ffmpeg/libavutil:$(pwd)/ffmpeg/libavformat:$(pwd)/ffmpeg/libavcodec:$(pwd)/ffmpeg/libswresample

tar xvzf opencv.tgz && cd opencv/opencv-4.x/ && mkdir build
cd ../..
export PKG_CONFIG_PATH=$(pwd)/ffmpeg/libavfilter:$(pwd)/ffmpeg/libavdevice:$(pwd)/ffmpeg/libswscale:$(pwd)/ffmpeg/libavutil:$(pwd)/ffmpeg/libavformat:$(pwd)/ffmpeg/libavcodec:$(pwd)/ffmpeg/libswresample
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.8


cd opencv/opencv-4.x/build/

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PACKAGE=OFF \
-D BUILD_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.6 \
-D WITH_CUDA=ON \
-D WITH_CUBLAS=ON \
-D WITH_CUFFT=ON \
-D WITH_NVCUVID=ON \
-D WITH_IPP=OFF \
-D WITH_V4L=ON \
-D WITH_1394=OFF \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=OFF \
-D WITH_EIGEN=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_JAVA=OFF \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D OPENCV_SKIP_PYTHON_LOADER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/home/k100/docker_space/base/opencv/opencv_contrib-4.x/modules \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=8.6 \
-D CUDA_ARCH_PTX=8.6 \
-D CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.8.4.1 \
-D PYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.8 \
-D PYTHON3_NUMPY_INCLUDE_DIRS=$CONDA_PREFIX/lib/python3.8/site-packages/numpy/core/include/ \
-D PYTHON3_PACKAGES_PATH=$CONDA_PREFIX/lib/python3.8/site-packages \
-D PYTHON3_LIBRARY=$CONDA_PREFIX/lib/python3.8 \
-D PYTHON_EXECUTABLE:FILEPATH=$CONDA_PREFIX/bin/python3 \
-D PYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python3 \
-D PYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python3 \
-D CUDNN_INCLUDE_DIR=/usr/local/cuda-11.6/include ..

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

$ sudo vim /etc/ld.so.conf
/usr/local/ffmpeg/lib/
$ sudo ldconfig

$ make -j$(nproc)
$ make install



# ------------------------------------------------------------------------------

$ 
conda create --name motion --clone cudacv
conda activate motion
#conda install pytorch==1.10 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl
pip install torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl
#pip install typing_extensions==4.5.0
# pip install opencv-python-headless

pip install openmim
mim install mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl 

pip install scipy
pip install fvcore
pip install seaborn
pip install pytorch-lightning==1.1.4
pip install einops
pip install joblib
pip install timm
pip install thop
pip install yacs==0.1.8
$ cd ~/Code/main/ultralytics
pip install -e .
pip uninstall opencv-python
pip install setuptools==59.5.0

```

## Dockerfile.v1
```dockerfile!
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
```

## Dockerfile.juri
[git python-opencv-gstreamer-docker-container ](https://github.com/juri117/python-opencv-gstreamer-docker-container/blob/main/Dockerfile)
```dockerfile!
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
```
## Dockerfile.gs
[Ubuntu20.04 + Gstreamer + opencv-python](https://blog.csdn.net/alan_yunshan/article/details/128370019)
```shell!
docker run -ti -d \
 --runtime=nvidia \
 --name=build_env \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 /bin/bash
```
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

## install meson
RUN  python3 -m pip install pip --upgrade
RUN  pip3 install meson 
  
## install ffmpeg  for opencv lib
WORKDIR /tmp
COPY ffmpeg.tgz ./
RUN tar xvzf ffmpeg.tgz && cd ffmpeg/ &&  ./configure --enable-shared --prefix=/usr/local/ffmpeg && make && make install

WORKDIR /tmp
RUN export PKG_CONFIG_PATH=$(pwd)/ffmpeg/libavfilter:$(pwd)/ffmpeg/libavdevice:$(pwd)/ffmpeg/libswscale:$(pwd)/ffmpeg/libavutil:$(pwd)/ffmpeg/libavformat:$(pwd)/ffmpeg/libavcodec:$(pwd)/ffmpeg/libswresample
RUN echo $PKG_CONFIG_PATH
RUN rm -rf ffmpeg.tgz

## install gstreamer
RUN  git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git
WORKDIR /gstreamer
RUN  meson setup --prefix /usr build -Dgst-plugins-bad:nvcodec=enabled
RUN  ninja -C build install
RUN  gst-inspect-1.0 --version
RUN  gst-inspect-1.0 -a |grep nvh264
RUN  gst-inspect-1.0 nvcodec

## install opencv lib
RUN apt-get install -y build-essential \
    cmake git python3-dev python3-numpy \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev libgtk2.0-dev \
    libcanberra-gtk-module libpng-dev \
    libjpeg-dev libopenexr-dev libtiff-dev \
    libwebp-dev libopencv-dev x264 \
    libx264-dev libssl-dev 

WORKDIR /tmp
RUN tar xvzf opencv.tgz && \
    cd opencv/opencv-4.x/ && \
    mkdir build
##RUN cd opencv && git checkout 4.5.4 && \
##	git submodule update --recursive --init && \
##    mkdir build && cd build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D PYTHON_EXECUTABLE=$(which python3) \
	-D BUILD_opencv_python2=OFF \
	-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
	-D PYTHON3_EXECUTABLE=$(which python3) \
	-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
	-D WITH_GSTREAMER=ON \
	-D BUILD_EXAMPLES=ON .. && \
	make -j$(nproc) && \
	make install && \
	ldconfig
```

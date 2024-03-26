# How to work?

## X86 CPU build
* start build image & run container
```
## WARNING : do not use root
sudo sh run_cnt_x86_cpu.sh
sudo docker image
## gst_python3_opencv:x86_cpu

sudo docker ps
## show container name: gst-cnt-cpu
sudo docker exec -ti gst-cnt-cpu bash

## into container
cd /workdir/main
python3 testRTSP.py
```
## X86 GPU build
* Before start , make sure you have those files
```
cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
ffmpeg.tgz gmp-6.2.1.tar.lz
gsl-latest.tar.gz
gstreamer-1.19.2.tar.xz
gst-plugins-base-1.19.2.tar.xz
gst-plugins-good-1.19.2.tar.xz
gst-plugins-bad-1.19.2.tar.xz
gst-plugins-ugly-1.19.2.tar.xz
opencv.tgz
```
* start build image & run container
```
## WARNING : do not use root
sudo sh run_cnt_x86_gpu.sh
sudo docker image
## gst_python3_opencv:x86_gpu

sudo docker ps
## show container name: gst-cnt-gpu
sudo docker exec -ti gst-cnt-gpu bash

## into container
cd /workdir/main
python3 test_opencv_gst_gpu.py
```

# How to work?

## X86 CPU build

```
## WARNING : do not use root
sudo sh run_cnt_x86_cpu.sh
sudo docker ps

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
```
## WARNING : do not use root
sudo sh run_cnt_x86_gpu.sh
sudo docker ps

sudo docker exec -ti gst-cnt-gpu bash

## into container
cd /workdir/main
python3 testRTSP.py
```

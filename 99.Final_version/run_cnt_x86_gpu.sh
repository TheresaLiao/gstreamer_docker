docker stop gst-cnt-gpu
docker rm gst-cnt-gpu

docker build -t gst_python3_opencv:x86_gpu -f Dockerfile.x86_gpu .

xhost +local:docker
docker run -ti -d \
       -w /home \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY=$DISPLAY \
       -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
       --name gst-cnt-gpu \
       gst_python3_opencv:x86_gpu bash

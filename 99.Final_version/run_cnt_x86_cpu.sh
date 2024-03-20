docker stop gst-cnt-cpu
docker rm gst-cnt-cpu

docker build -t gst_python3_opencv:x86_cpu -f Dockerfile.x86_cpu .

xhost +local:docker
docker run -ti -d \
       -w /workdir \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY=$DISPLAY \
       --name gst-cnt-cpu \
       gst_python3_opencv:x86_cpu bash

#-v ${PWD}:/workdir \

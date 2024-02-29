docker stop opencv-gstreamer-test
docker rm opencv-gstreamer-test


docker build -t opencv-gstreamer .
xhost +local:docker

docker run -it -d \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name opencv-gstreamer-test \
    opencv-gstreamer bash


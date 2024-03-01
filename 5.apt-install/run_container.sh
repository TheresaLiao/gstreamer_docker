docker stop gt-container-gpu
docker rm gt-container-gpu

#docker build -t gt-container:gpu .

xhost +local:docker
docker run -ti -d \
       -v ${PWD}:/workdir \
       -w /workdir \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY=$DISPLAY \
       --name gt-container-gpu \
       gt-container:gpu bash

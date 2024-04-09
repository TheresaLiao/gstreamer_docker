sh ./run_cnt_x86_gpu_12_0.sh

container_name=gst-cnt-gpu_torch_12_0
image_name=gst_python3_opencv:x86_gpu_torch_12_0
dockerfile_name=Dockerfile.x86_gpu_torch_12_0

docker stop $container_name
docker rm $container_name

docker build -t $image_name -f $dockerfile_name .

xhost +local:docker
docker run -ti -d \
       -w /home \
       -v /home/samba/raw_result:/home/samba/raw_result \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY=$DISPLAY \
       -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
       --name $container_name \
       $image_name bash

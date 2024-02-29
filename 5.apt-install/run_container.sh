docker stop gt-container
docker rm gt-container

docker build -t gt-container:1.0 .
docker run -ti -d \
       -v ${PWD}:/workdir \
       -w /workdir \
       --name gt-container \
       gt-container:1.0 bash

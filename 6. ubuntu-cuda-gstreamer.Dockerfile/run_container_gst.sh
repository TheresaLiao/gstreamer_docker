docker build -t gstreamer-builder:1.0 -f Dockerfile.gstreamer-builder .
docker build -t gstreamer-builder:2.0 -f Dockerfile.gstreamer-final .
docker run -it -d  gstreamer-builder:2.0 bash

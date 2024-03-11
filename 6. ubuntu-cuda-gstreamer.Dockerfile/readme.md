# Test
* build contaier
* ref : https://gist.github.com/m1k1o/28c73fc15cd1fba59b73364c3b7a5d0a
```
sh run_container_gst.sh
docker exec -ti <container_name> bash

## work
gst-launch-1.0 -v rtspsrc location=rtsp://admin:admin@10.1.1.202:554/profile1 latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! cudaupload  ! cudadownload ! appsink

## Not work
root@8cc513b1a56c:/# gst-launch-1.0 -v rtspsrc location=rtsp://admin:admin@10.1.1.202:554/profile1 latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! cudaupload ! cudaconvert ! cudadownload ! appsink
0:00:00.229079255   156 0x55d830c18b90 ERROR           GST_PIPELINE subprojects/gstreamer/gst/parse/grammar.y:570:gst_parse_element_make: no element "cudaconvert"
0:00:00.229100309   156 0x55d830c18b90 ERROR           GST_PIPELINE subprojects/gstreamer/gst/parse/grammar.y:1264:priv_gst_parse_yyparse: link has no sink [source=@0x55d830efc7b0]
0:00:00.229125510   156 0x55d830c18b90 ERROR           GST_PIPELINE subprojects/gstreamer/gst/parse/grammar.y:1264:priv_gst_parse_yyparse: link has no source [sink=@0x55d830d108d0]
WARNING: erroneous pipeline: no element "cudaconvert"
```

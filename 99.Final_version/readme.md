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

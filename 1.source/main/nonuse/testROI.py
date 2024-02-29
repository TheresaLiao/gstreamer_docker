import os
f = open("ROI_setup/cams_Sort.txt")
camsSort = []
for line in f.readlines():
    camsSort.append(line[:-1])
f.close()

ROI_all_cams = []
for cam in camsSort:
    if os.path.exists("ROI_setup/"+cam):
        f = open("ROI_setup/"+cam)
        ROI_EachCam = []
        for line in f.readlines():
            x,y,w,h = line.split(" ")
            h = h[:-1]
            x,y,w,h = int(x),int(y),int(w),int(h)
            ROI_EachCam.append([x,y,w,h])
        f.close()
        if len(ROI_EachCam):
            ROI_all_cams.append(ROI_EachCam)
        else:
            ROI_all_cams.append(None)
    else:
        print("ROI_setup/"+cam, " not exist....")
        ROI_all_cams.append(None)



import numpy as np
import torch
import json
from shapely.geometry import Polygon, box
from cfg_testInteractive import *
import datetime,time
import cv2
import os


def calculate_coverage(bbox1, bbox2):
    """
    计算bbox1被bbox2覆盖的百分比
    """
    
    """
    计算两个边界框的交集面积
    """
    x1_int = max(bbox1[0], bbox2[0])
    y1_int = max(bbox1[1], bbox2[1])
    x2_int = min(bbox1[2], bbox2[2])
    y2_int = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2_int - x1_int + 1) * max(0, y2_int - y1_int + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)

    coverage_percentage = intersection_area / bbox1_area
    return coverage_percentage
    
def kpt2bbox(kptSeq):
    x_min,x_max = kptSeq[-1,:,0].min().item(),kptSeq[-1,:,0].max().item()
    y_min,y_max = kptSeq[-1,:,1].min().item(),kptSeq[-1,:,1].max().item()
    return (x_min,y_min,x_max,y_max)

def xywhTOxyxy(bbox):
    w_half,h_half = bbox[2]/2,bbox[3]/2
    return (bbox[0]-w_half,bbox[1]-h_half,bbox[0]+w_half,bbox[1]+h_half)

def bbox_shortest_distance(bbox1, bbox2,mode = 'xywh'):
    if mode == 'xywh':
        print(bbox1)
        bbox1 = xywhTOxyxy(bbox1)
        bbox2 = xywhTOxyxy(bbox2)
        print(bbox1)
    value = None
    if bbox1[3] < bbox2[1]:
        value = bbox2[1] - bbox1[3]
    elif bbox1[1] > bbox2[3]:
        value = bbox1[1] - bbox2[3]
    elif bbox1[2] < bbox2[0]:
        value = bbox2[0] - bbox1[2]
    elif bbox1[0] > bbox2[2]:
        value = bbox1[0] - bbox2[2]
    else:
        value = 0
    return value


def save4OffLineReID():
    '''
    _____________________________________________________________________
     [TODO] :
        * Find a way to save img and infomation effectively
            --> h5py   (V)
            --> pickle (X)
        
     [SAVE information] : 
        * Crop image (hight kpt visibility)
        * bbox
        * motion
        * confidence
        * keypoint Seq ?
        * snapshot path

     [INPUT] : 
        * image (Origin without draw)
        * bbox (x1,y1,x2,y2)
        * scale (ex: (1280,720))
        * keypoint Seq (ex : shape is (60,40,17,3))
        * snapshot path


    _____________________________________________________________________

    '''



    return 0


def rawDetect2Info(actIndexes=None, actScores=None,
                   keypointsSeqs=None, bboxesSort=None,
                   belongLine=None, trackIDss=None,
                   threshold_Neighbor=1, imgs=None,
                   singleDangerous_cls = singleDangerous_cls,
                   interactiveDangerous_cls = interactiveDangerous_cls,
                   color = (0,255,0), writeImg=True ):
                   
    current_date = str(datetime.date.today())
    current_datetime =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    dataIdxes = defaultdict(lambda: [])
    print("belongLine : ", belongLine)

    for resultID,belongLineIndex in enumerate(belongLine):
        dataIdxes[belongLineIndex].append(resultID)
    print("dataIdxes[belongLineIndex] : ", dataIdxes[belongLineIndex])

    eventDicts =[]
    for belongLineIndex in dataIdxes:
        check_interactive = False

        resultIDX_in_line = np.array(dataIdxes[belongLineIndex])
        actIndexesInLine = actIndexes[resultIDX_in_line]
        actScoresInLine = actScores[resultIDX_in_line]
        keypointsSeqsInLine = keypointsSeqs[resultIDX_in_line]
        bboxesSortInLine = bboxesSort[resultIDX_in_line]
        trackIDssInLine = trackIDss[resultIDX_in_line]

        isinnteractiveDangerous = np.isin(actIndexesInLine, interactiveDangerous_cls_keys)
        isinsingleDangerous = np.isin(actIndexesInLine, singleDangerous_cls_keys)
        if np.any(isinnteractiveDangerous):
            if len(keypointsSeqsInLine)>1:
                check_interactive = True
            else:
                print("Innteractive Dangerous exist but only one person")
        isinsingleDangerous = np.any(isinsingleDangerous)
            
        img = imgs[belongLineIndex]
        if isinsingleDangerous or np.any(isinnteractiveDangerous):
            img = cv2.resize(img,(write_w,write_h))
        for idx,(actI,actS,kptSeq,bbox,tidd)in enumerate(zip(actIndexesInLine,actScoresInLine,keypointsSeqsInLine,bboxesSortInLine,trackIDssInLine)):
            if TieCode:
                actI = actionMatch2ORGcls[actI]
            write_img = None
            
            if actS >= PUBLIC_THRESHOLD:  
                if (actI in singleDangerous_cls_values or actI in interactiveDangerous_cls_values) and not check_interactive:
                    p1,p2 = None,None
                    if writeImgResize:
                        w_half = (bbox[2]*(write_w/video_w))/2
                        h_half = (bbox[3]*(write_h/video_h))/2
                        xc = (bbox[0]*(write_w/video_w))
                        yc = (bbox[1]*(write_h/video_h))
                        p1 = (int(xc-w_half),int(yc-h_half)) 
                        p2 = (int(xc+w_half),int(yc+h_half)) 
                    else:
                        w_half = bbox[2]/2
                        h_half = bbox[3]/2
                        xc = bbox[0]
                        yc = bbox[1]
                        p1 = (int(xc-w_half),int(yc-h_half)) 
                        p2 = (int(xc+w_half),int(yc+h_half)) 

                    pc = (int(bbox[0]*(eventServer_expect_w/video_w)),int(bbox[1])*(eventServer_expect_h/video_h))

                    write_img = img.copy()
                    cv2.rectangle(write_img,p1,p2,color,1)

                    imgName = f"/home/samba/raw_result/{current_date}/camIndex_{belongLineIndex}/camIndex_{belongLineIndex}_{current_datetime}_ACT_{actI}_{actS}.png"
                    shapshotPath = imgName.replace("/home/samba/raw_result","/images")
                    if writeImg:
                        subPath = os.path.dirname(imgName)
                        subsubPath = os.path.dirname(subPath)
                        if not os.path.exists(subsubPath):
                            os.mkdir(subsubPath)
                        if not os.path.exists(subPath):
                            os.mkdir(subPath)
                        cv2.imwrite(imgName,write_img)
                        print("Write image : ",imgName)
                    eventDicts.append({
                                        "start_time":0, "user_id":9487,
                                        "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(tidd).zfill(7)}",
                                        "event_action": int(actI),
                                        "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                                        "confidence":int(actS),"prediction_status":0,
                                        "event_action_id_2nd": 0,"confidence_2nd":0,"event_action_id_3rd": 0,"confidence_3rd":0,
                                        "snapshot":shapshotPath,
                                        "center_x":pc[0],
                                        "center_y":pc[1],
                                        })
                
                elif actI in interactiveDangerous_cls_values and check_interactive:
                    real_dangerous = False
                    bboxes_check = bboxesSortInLine.copy()
                    bboxes_check = np.delete(bboxes_check,idx,axis=0)
                    for bbox_check in bboxes_check:
                        short_dis = bbox_shortest_distance(bbox_check,bbox)
                        print(f"[short_dis] : {short_dis} is large than {threshold_Neighbor}(thres) --> bboxes are : {bbox_check} and {bbox}")
                        if short_dis <= threshold_Neighbor:
                            real_dangerous = True
                            break                

                    print(f"actI : {actI} --> (real_dangerous : {real_dangerous})")
                    if real_dangerous: # ex: 64 --> 129
                        if actI in interactiveDangerous_cls_transfer_key:
                            actI = interactiveDangerous_cls_transfer[actI]
                        else:
                            print(f"{actI} not in defaultdict interactiveDangerous_cls")

                    # actS = actScoresInLine[org_idx]
                    imgName = f"/home/samba/raw_result/{current_date}/camIndex_{belongLineIndex}/camIndex_{belongLineIndex}_{current_datetime}_ACT_{actI}_{actS}.png"
                    shapshotPath = imgName.replace("/home/samba/raw_result","/images")

                    p1,p2 = None,None
                    if writeImgResize:
                        w_half = (bbox[2]*(write_w/video_w))/2
                        h_half = (bbox[3]*(write_h/video_h))/2
                        xc = (bbox[0]*(write_w/video_w))
                        yc = (bbox[1]*(write_h/video_h))
                        p1 = (int(xc-w_half),int(yc-h_half)) 
                        p2 = (int(xc+w_half),int(yc+h_half)) 
                    else:
                        w_half = bbox[2]/2
                        h_half = bbox[3]/2
                        xc = bbox[0]
                        yc = bbox[1]
                        p1 = (int(xc-w_half),int(yc-h_half)) 
                        p2 = (int(xc+w_half),int(yc+h_half)) 
                    pc = (int(bbox[0]*(eventServer_expect_w/video_w)),int(bbox[1])*(eventServer_expect_h/video_h))
                    write_img = img.copy()
                    cv2.rectangle(write_img,p1,p2,color,1)
                    if writeImg:
                        subPath = os.path.dirname(imgName)
                        subsubPath = os.path.dirname(subPath)
                        if not os.path.exists(subsubPath):
                            os.mkdir(subsubPath)
                        if not os.path.exists(subPath):
                            os.mkdir(subPath)
                        cv2.imwrite(imgName,write_img)
                        print("Write image : ",imgName)

                    eventDicts.append({
                        "start_time":0, "user_id":9487,
                        "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(tidd).zfill(7)}",
                        # "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(trackIDssInLine[org_idx]).zfill(7)}",
                        "event_action": int(actI),
                        "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                        "confidence":int(actS),"prediction_status":0,
                        "event_action_id_2nd": 0,"confidence_2nd":0,
                        "event_action_id_3rd": 0,"confidence_3rd":0,
                        "snapshot":shapshotPath,
                        "center_x":pc[0],
                        "center_y":pc[1],
                        })
                            
    return eventDicts



def rawDetect2Info_v1(actIndexes=None,
                   actScores=None,
                   keypointsSeqs=None,
                   bboxesSort=None,
                   belongLine=None,
                   trackIDss=None,
                   threshold_Neighbor=10,
                   imgs=None,
                   singleDangerous_cls = singleDangerous_cls,
                   interactiveDangerous_cls = interactiveDangerous_cls,
                   color = (0,255,0),
                   writeImg=True,
                  ):
    current_date = str(datetime.date.today())
    current_datetime =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    dataIdxes = defaultdict(lambda: [])
    print("belongLine : ", belongLine)

    for resultID,belongLineIndex in enumerate(belongLine):
        dataIdxes[belongLineIndex].append(resultID)
        #print(dataIdxes)
    print("dataIdxes[belongLineIndex] : ", dataIdxes[belongLineIndex])

    eventDicts =[]
    for belongLineIndex in dataIdxes:
        check_interactive = False

        resultIDX_in_line = np.array(dataIdxes[belongLineIndex])
        actIndexesInLine = actIndexes[resultIDX_in_line]
        actScoresInLine = actScores[resultIDX_in_line]
        keypointsSeqsInLine = keypointsSeqs[resultIDX_in_line]
        bboxesSortInLine = bboxesSort[resultIDX_in_line]
        trackIDssInLine = trackIDss[resultIDX_in_line]

        isinnteractiveDangerous = np.isin(actIndexesInLine, interactiveDangerous_cls_keys)
        isinsingleDangerous = np.isin(actIndexesInLine, singleDangerous_cls_keys)
        if np.any(isinnteractiveDangerous):
            if len(keypointsSeqsInLine)>1:
                check_interactive = True
            else:
                print("Innteractive Dangerous exist but only one person")
        isinsingleDangerous = np.any(isinsingleDangerous)
            
        img = imgs[belongLineIndex]
        if isinsingleDangerous or np.any(isinnteractiveDangerous):
            img = cv2.resize(img,(write_w,write_h))
        bboxes_dangerous = [] ; dangerous_idxes = []
        for idx,(actI,actS,kptSeq,bbox,tidd)in enumerate(zip(actIndexesInLine,actScoresInLine,keypointsSeqsInLine,bboxesSortInLine,trackIDssInLine)):
            if TieCode:
                actI = actionMatch2ORGcls[actI]
            write_img = None
            if actS >= PUBLIC_THRESHOLD:  
                if (actI in singleDangerous_cls_values or actI in interactiveDangerous_cls_values) and not check_interactive:
                    p1,p2 = None,None
                    if writeImgResize:
                        w_half = (bbox[2]*(write_w/video_w))/2
                        h_half = (bbox[3]*(write_h/video_h))/2
                        xc = (bbox[0]*(write_w/video_w))
                        yc = (bbox[1]*(write_h/video_h))
                        p1 = (int(xc-w_half),int(yc-h_half)) 
                        p2 = (int(xc+w_half),int(yc+h_half)) 
                    else:
                        w_half = bbox[2]/2
                        h_half = bbox[3]/2
                        xc = bbox[0]
                        yc = bbox[1]
                        p1 = (int(xc-w_half),int(yc-h_half)) 
                        p2 = (int(xc+w_half),int(yc+h_half)) 

                    pc = (int(bbox[0]*(eventServer_expect_w/video_w)),int(bbox[1])*(eventServer_expect_h/video_h))

                    write_img = img.copy()
                    cv2.rectangle(write_img,p1,p2,color,1)

                    imgName = f"/home/samba/raw_result/{current_date}/camIndex_{belongLineIndex}/camIndex_{belongLineIndex}_{current_datetime}_ACT_{actI}_{actS}.png"
                    shapshotPath = imgName.replace("/home/samba/raw_result","/images")
                    if writeImg:
                        subPath = os.path.dirname(imgName)
                        subsubPath = os.path.dirname(subPath)
                        if not os.path.exists(subsubPath):
                            os.mkdir(subsubPath)
                        if not os.path.exists(subPath):
                            os.mkdir(subPath)
                        cv2.imwrite(imgName,write_img)
                        print("Write image : ",imgName)
                    eventDicts.append({
                                        "start_time":0, "user_id":9487,
                                        "uu_id":f"{str(belongLineIndex+1).zfill(2)}{str(tidd).zfill(7)}",
                                        "event_action": int(actI),
                                        "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                                        "confidence":int(actS),"prediction_status":0,
                                        "event_action_id_2nd": 0,"confidence_2nd":0,"event_action_id_3rd": 0,"confidence_3rd":0,
                                        "snapshot":shapshotPath,
                                        "center_x":pc[0],
                                        "center_y":pc[1],
                                        })
                
                elif actI in interactiveDangerous_cls_values and check_interactive:
                    bboxes_dangerous.append(bbox)
                    dangerous_idxes.append(idx)
                
        for idxx,bbox_dangerous in enumerate(bboxes_dangerous):
            real_dangerous = False
            # org_idx = dangerous_idxes[idxx]
            
            bboxes_check = bboxesSortInLine.copy()
            bboxes_check = np.delete(bboxes_check,idxx,axis=0)
            
            for bbox_check in bboxes_check:
                short_dis = bbox_shortest_distance(bbox_check,bbox_dangerous)
                print(f"[short_dis] : {short_dis} is large than {threshold_Neighbor}(thres) --> bboxes are : {bbox_check} and {bbox_dangerous}")
                
                
                if short_dis <= threshold_Neighbor:
                    real_dangerous = True
                    break
         
                
            bbox = bbox_dangerous
            # actI = actIndexesInLine[org_idx]
            # if TieCode:
                # actI = actionMatch2ORGcls[actI]
            # tiddd = str(trackIDssInLine[org_idx])
            tiddd = str(tidd)

            print("actI : ",actI)
            print("real_dangerous : ",real_dangerous)

            if real_dangerous: # ex: 64 --> 129

                if actI in interactiveDangerous_cls_transfer_key:
                    actI = interactiveDangerous_cls_transfer[actI]
                else:
                    print(f"{actI} not in defaultdict interactiveDangerous_cls")

            # actS = actScoresInLine[org_idx]
            imgName = f"/home/samba/raw_result/{current_date}/camIndex_{belongLineIndex}/camIndex_{belongLineIndex}_{current_datetime}_ACT_{actI}_{actS}.png"
            shapshotPath = imgName.replace("/home/samba/raw_result","/images")

            p1,p2 = None,None
            if writeImgResize:
                w_half = (bbox[2]*(write_w/video_w))/2
                h_half = (bbox[3]*(write_h/video_h))/2
                xc = (bbox[0]*(write_w/video_w))
                yc = (bbox[1]*(write_h/video_h))
                p1 = (int(xc-w_half),int(yc-h_half)) 
                p2 = (int(xc+w_half),int(yc+h_half)) 
            else:
                w_half = bbox[2]/2
                h_half = bbox[3]/2
                xc = bbox[0]
                yc = bbox[1]
                p1 = (int(xc-w_half),int(yc-h_half)) 
                p2 = (int(xc+w_half),int(yc+h_half)) 
            pc = (int(bbox[0]*(eventServer_expect_w/video_w)),int(bbox[1])*(eventServer_expect_h/video_h))
            write_img = img.copy()
            cv2.rectangle(write_img,p1,p2,color,1)
            if writeImg:
                subPath = os.path.dirname(imgName)
                subsubPath = os.path.dirname(subPath)
                if not os.path.exists(subsubPath):
                    os.mkdir(subsubPath)
                if not os.path.exists(subPath):
                    os.mkdir(subPath)
                cv2.imwrite(imgName,write_img)
                print("Write image : ",imgName)

            eventDicts.append({
                "start_time":0, "user_id":9487,
                "uu_id":f"{str(belongLineIndex+1).zfill(2)}{tiddd.zfill(7)}",
                "event_action": int(actI),
                "status":3,"group_id":'G01',"location_id":belongLineIndex+1,
                "confidence":int(actS),"prediction_status":0,
                "event_action_id_2nd": 0,"confidence_2nd":0,
                "event_action_id_3rd": 0,"confidence_3rd":0,
                "snapshot":shapshotPath,
                "center_x":pc[0],
                "center_y":pc[1],
                })
                    
    return eventDicts


def ROIs_define_showUse(video_list):
    infos = {}
    for video_name in video_list:
        infos[video_name] = None

    f = open("ROI_setup/ROI_show_use.txt")
    for line in f.readlines():
        info = json.loads(line[:-1])
        rtsp_info = info['RTSP']
        poly_info = info['poly']
        if rtsp_info in infos.keys():
            infos[rtsp_info] = poly_info
    f.close()
    ROI_all_cams = []
    for video_name in video_list:
        if  video_name in infos.keys():
            ROI_all_cams.append(Polygon(infos[video_name]))
            
        else:
            ROI_all_cams.append(None)
    return ROI_all_cams

def sources_read(vidcaps,sourcesList,imgs):
    for video_index,source in enumerate(sourcesList):
        ret, image = vidcaps[video_index].read()
        if not ret and cycle:
            vidcaps[video_index].release()
            time.sleep(0.001)
            if "rtsp" in video_list[video_index]:
                if use_gst:
                    vidcaps[video_index] = cap.VideoCapture(video_list[video_index],0,0,0)
                else:
                    vidcaps[video_index] = cap.VideoCapture(video_list[video_index])
            else:
                vidcaps[video_index] = cv2.VideoCapture(video_list[video_index])
            continue
        elif not ret and not cycle:
            print("video Finished ...............")
        else:
            imgs[video_index] = image
    return imgs

def multi_video_load(sourcesList):
    vidcaps = []
    for video_index,source in enumerate(sourcesList):
        if "rtsp" in video_list[video_index]:
            if use_gst:
                vidcaps.append(cap.VideoCapture(source,0,0,0))
            else:
                vidcaps.append(cap.VideoCapture(source))

        else:
            vidcaps.append(cv2.VideoCapture(source))
    return vidcaps

def motion_result_postprocess(results): # cpu result
    actIndexes,actScores = np.argmax(results,axis=1),np.max(results,axis=1)
    actScores = (100*actScores).astype(int)

    top10_indices = np.argpartition(-results, 9, axis=1)[:, :9]
    top10_values = np.array([row[row_indices] for row, row_indices in zip(results, top10_indices)])
    sorted_order = np.argsort(-top10_values, axis=1)
    top10_values_sorted = [row[row_order] for row, row_order in zip(top10_values, sorted_order)]
    top10_indices_sorted = [row_indices[row_order] for row_indices, row_order in zip(top10_indices, sorted_order)]

    return actIndexes,actScores,top10_indices_sorted,top10_values_sorted



# p1 = (int(bbox[0]*(write_w/video_w)),int(bbox[1]*(write_h/video_h)))
# p2 = (int(bbox[2]*(write_w/video_w)),int(bbox[3]*(write_h/video_h)))
# pc = (int((eventServer_expect_w/write_h)*(p1[0]+p2[0])/2),
#       int((eventServer_expect_h/write_h)*(p1[1]+p2[1])/2))
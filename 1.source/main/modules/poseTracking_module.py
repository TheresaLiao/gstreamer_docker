import numpy as np
import cv2
import json
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import deque
import time
from copy import deepcopy
import pprint
pprint = pprint.PrettyPrinter(depth=6).pprint

class KalmanBoxTracker():
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # z: [x_center,y_center,area,ratio]
        # x: [x_center,y_center,area,ratio, D(x_center),D(y_center),D(area)]
        # F:        transition matrix
        # H:       measurement matrix
        # P:        covariance matrix
        # Q:     process noise matrix
        # R: measurement noise matrix
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],

                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.P[ 4:, 4:] *= 1e3 # give high uncertainty to the unobservable initial velocities
        self.kf.P          *= 1e1
        self.kf.Q[-1 ,-1 ] *= 1e-2
        self.kf.Q[ 4:, 4:] *= 1e-2
        self.kf.R[ 2:, 2:] *= 1e1

        self.kf.x[:4] = self.bbox2z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1




        # To predict every time, but to update when successful association
        #               age: increasing every time
        #              hits: increasing every "update" 
        # time_since_update: increasing every "predict" but resetted when "update"
        #        hit_streak: increasing every "update"  but resetted when failed association 
        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_since_update = 0
        self.history = []

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.z2bbox(self.kf.x[:4])

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.kf.update(self.bbox2z(bbox))
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.history = []

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.kf.x[6]+self.kf.x[2] <= 0:
            self.kf.x[6] = 0.
        self.kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def bbox2z(self,bbox):
        """
        Takes a bounding box in the form [left,top,right,bottom] and returns z in the form
        [x_center,y_center,area,ratio]
        """
        bbox = np.float32(bbox)
        width    = bbox[2]-bbox[0]
        height   = bbox[3]-bbox[1]
        x_center = bbox[0]+width/2.
        y_center = bbox[1]+height/2.
        area     = width*height
        ratio    = float(width)/float(height)
        return np.float32([x_center,y_center,area,ratio]).reshape((4,1))

    def z2bbox(self,z_kf):
        """
        Takes a bounding box in the centre form [x_center,y_center,area,ratio] and returns it in the form
        [left,top,right,bottom]
        """
        width  = np.sqrt(z_kf[2]*z_kf[3]) # (wh)(w/h) = w^2
        height = z_kf[2]/width # w/(w/h) = h
        left   = z_kf[0]-width/2.
        top    = z_kf[1]-height/2.
        right  = z_kf[0]+width/2.
        bottom = z_kf[1]+height/2.
        return np.float32([left,top,right,bottom]).reshape((1,4))




class Sort():

    def __init__(self,
                 height_max,
                 width_max,
                 categories=[], # default, [], is not to ingore anything
                 record_absence=False,  # trajectories_opt
                 output_absence=False,  # json
                 IOU_threshold=0.1,
                 max_miss=5,   # allow miss 4 frames
                 min_hits=3,   # yolo detect should > 3 times, then start track  
                 max_history=None,
                 camIndex = 0,
                 
                 ):

        """
        Set parameters for SORT
        """
        self.height_max = height_max
        self.width_max = width_max
        self.categories = set( categories  if (type(categories) is list) else  [categories] )
        self.record_absence = record_absence
        self.output_absence = output_absence
        self.IOU_threshold = IOU_threshold
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.max_history = max_history

        self.frame_count = 0
        self._json_track_present = {}
        self.trackers = []
        self._info = {}
        self.camIndex = camIndex
        
    def tracking(self,json_yolo):
        self.death = {}
        # if self.categories:
        #     json_yolo = self._filter_category(json_yolo)

        # CONVERT yolo results in json format (json_yolo) TO matching format (dets)
        # dets = self._json_yolo_to_detections(json_yolo)
        # trks = self._update_trk(dets)
        trks = self._update_trk(json_yolo)
        #print(len(trks))

        # UPDATE tracking results (trks) TO intermediate information (_info)
        self._update_info(trks, json_yolo)

        # CONVERT intermediate information (_info) TO tracking results in json format (json_track)
        json_track = self._get_json_track()

        self.frame_count += 1

        return json_track

    def _filter_category(self,json_yolo):
        json_yolo_filtered = deepcopy(json_yolo)
        for i,bbox in reversed(list(enumerate(json_yolo_filtered))):
            if not self.categories.intersection( set(bbox["objectTypes"]) ):
                json_yolo_filtered.pop(i)
        return json_yolo_filtered

    def _update_trk(self,json_yolo):
        """
        Params:
          dets - a numpy array of detections in the format [[left,top,right,bottom,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        dets = []
        for det in json_yolo:
            dets.append([det["objectPicX"],\
                            det["objectPicY"],\
                            det["objectPicX"]+det["objectWidth"],\
                            det["objectPicY"]+det["objectHeight"]])
        dets = np.array(dets)



        # get predicted locations from existing trackers.
        trks = np.empty((len(self.trackers),4))
        to_delete = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_delete.append(t)
        for t in reversed(to_delete):
            self.trackers.pop(t) # TODO pop
            exit() # TODO exit
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        matches, unmatched_dets, unmatched_trks = self._associate_dets_to_trks(dets,trks)

        # update matched trackers with assigned detections
        for t,tracker in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matches[np.where(matches[:,1]==t)[0],0]
                tracker.update(dets[d,:][0])

        # create and initialise new trackers for unmatched detections
        number_matches  = len(matches)
        number_trackers = len(self.trackers)
        matches = np.concatenate((matches,np.empty((len(unmatched_dets),2),dtype=int)))
        for i,j in enumerate(unmatched_dets):
            tracker = KalmanBoxTracker(dets[j,:]) 

            self.trackers.append(tracker)
            matches[number_matches+i,0] = j                 # det
            matches[number_matches+i,1] = number_trackers+i # trk
        self.matches_trk2det = { t:d  for d,t in matches }
        #self.matches_id = np.empty(len(dets),dtype=np.int32)
        #for index_det,index_trk in matches:
        #    self.matches_id[index_det] = self.trackers[index_trk].id

        # confirm trks that are detected in this frame
        trks = []
        number_trackers = len(self.trackers)
        for i,tracker in reversed(list(enumerate(self.trackers))):
            pos = tracker.get_state()[0]
            # detected in this frame and reaching number of criterion

            # left,top,right,bottom
            pos[0] = self._constrain_and_quantize(pos[0],0,self.width_max)
            pos[1] = self._constrain_and_quantize(pos[1],0,self.height_max)
            pos[2] = self._constrain_and_quantize(pos[2],0,self.width_max)
            pos[3] = self._constrain_and_quantize(pos[3],0,self.height_max)
            pos[2] = max(pos[0],pos[2])
            pos[3] = max(pos[1],pos[3])

            dead = ( tracker.time_since_update > self.max_miss ) | \
                   ( abs(pos[2]-pos[0]) == 0 ) | \
                   ( abs(pos[3]-pos[1]) == 0 )
            emerged = ( tracker.time_since_update == 0 ) & \
                      ( tracker.hit_streak >= min(self.min_hits,self.frame_count-1) ) & \
                      ( not dead )
            trks.append(np.concatenate((pos,
                                        [tracker.id,
                                         tracker.age,
                                         i, emerged, dead])).reshape(1,-1))

            # remove dead tracklet
            #if tracker.time_since_update > self.max_miss:
            if dead:
                self.trackers.pop(i)

        trks = np.concatenate(trks)  if len(trks) > 0 else  np.empty((0,9))
        return trks

    @staticmethod
    def get_IOUs(bboxes1,bboxes2):
        if len(bboxes1)==0 or len(bboxes1)==0:
            return np.zeros((len(bboxes1),len(bboxes2)),dtype=np.float32)

        x_overlap = np.minimum(bboxes1[:,None,2],bboxes2[None,:,2]) \
                  - np.maximum(bboxes1[:,None,0],bboxes2[None,:,0]) # right-left
        y_overlap = np.minimum(bboxes1[:,None,3],bboxes2[None,:,3]) \
                  - np.maximum(bboxes1[:,None,1],bboxes2[None,:,1]) # bottom-top
        area_i = np.where( (x_overlap>0) & (y_overlap>0) , x_overlap*y_overlap, 0.)
        area1 = (( bboxes1[:,2]-bboxes1[:,0] )*( bboxes1[:,3]-bboxes1[:,1] ))[:,None]
        area2 = (( bboxes2[:,2]-bboxes2[:,0] )*( bboxes2[:,3]-bboxes2[:,1] ))[None,:]
        return area_i/(area1+area2-area_i) # intersection over union

    def _associate_dets_to_trks(self,dets,trks):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_dets and unmatched_trks
               matches: 2
        unmatched_dets: 1
        unmatched_trks: 1
        """
        if len(trks) == 0:
            return ( np.empty((0,2),dtype=int),  # matches
                     np.arange(len(dets)),       # unmatched_dets
                     np.empty((0,1),dtype=int) ) # unmatched_trks

        IOU_matrix = self.get_IOUs(dets, trks)
        matched_indices = np.stack(linear_sum_assignment(-IOU_matrix), axis=1)

        unmatched_dets = list( set(range(len(dets))) - set(matched_indices[:,0]) )
        unmatched_trks = list( set(range(len(trks))) - set(matched_indices[:,1]) )

        #filter out matched with low IOU
        matched_indices_T = matched_indices.T
        filtered = ( IOU_matrix[ matched_indices_T[0] , matched_indices_T[1] ] >= self.IOU_threshold )
        matches = matched_indices[filtered,:]
        unmatched_dets += list(matched_indices_T[0][ np.logical_not(filtered) ])
        unmatched_trks += list(matched_indices_T[1][ np.logical_not(filtered) ])

        return ( matches,
                 np.array(unmatched_dets),
                 np.array(unmatched_trks)  )

    # def _json_yolo_to_detections(self,json_yolo):
    #     bboxes = []
    #     for bbox in json_yolo:
    #        bboxes.append([bbox["objectPicX"],\
    #                       bbox["objectPicY"],\
    #                       bbox["objectPicX"]+bbox["objectWidth"],\
    #                       bbox["objectPicY"]+bbox["objectHeight"]])
    #     bboxes = np.array(bboxes)
    #     return bboxes

    def _constrain_and_quantize(self,value,lower,upper):
        return int(round( min(max(value,lower),upper) ))

    def _update_info(self,trks,json_yolo):
        #print(self._info.keys())
        #for key,value in self._info.items():
        #    print(key, len(value.get('track')))
        for trk in trks:
            left,top,right,bottom = trk[:4]
            objectID  = int(trk[4])
            age       = int(trk[5])
            index_trk = int(trk[6])
            emerged   = int(trk[7])
            dead      = int(trk[8])
            objectWidth  = self._constrain_and_quantize(right-left,0,self.width_max)
            objectHeight = self._constrain_and_quantize(bottom-top,0,self.height_max)
            objectPicX   = self._constrain_and_quantize(      left,0,self.width_max)
            objectPicY   = self._constrain_and_quantize(       top,0,self.height_max)
            bbox = {"objectWidth"  : objectWidth,
                    "objectHeight" : objectHeight,
                    "objectPicX"   : objectPicX,
                    "objectPicY"   : objectPicY,
                    # "keypoints"    : keypoints,
                    # "keypointScore": keypointScore,
                    "frame"        : self.frame_count}

            #print(objectID, dead)
            if dead and objectID in self._info.keys():
                value = self._info.get(objectID)
                content = self._get_json_track_single(objectID,value)
                del content['objectID']
                self.death[objectID] = content
                del self._info[objectID] # TODO del
                # print(f"Succrssful del {objectID}")
            else:
                self._info.setdefault(objectID,{})

                self._info[objectID]['emerge'] = emerged
                if index_trk in self.matches_trk2det:
                    index_det = self.matches_trk2det[index_trk]
                    self._info[objectID]['detect'] = json_yolo[index_det]
                    self._info[objectID]['duration'] = int(age)
                    bbox["keypoints"] = json_yolo[index_det]["keypoints"]
                    # bbox["keypointScore"] = json_yolo[index_det]["keypointScore"]

                if emerged or self.record_absence:
                    q = []  if (self.max_history is None) else  deque(maxlen=self.max_history)
                    self._info[objectID].setdefault('track', q).append(bbox)

    def _get_json_track_single(self, key, value):
        json_track_single = value['detect']
        json_track_single['objectID'] = key
        json_track_single['duration'] = value['duration']
        # print(value.keys(),"valuevaluevaluevaluevaluevaluevaluevaluevaluevaluevaluevaluevalue")
        json_track_single['trajectories_opt'] = list(value.get('track',[]))
        return json_track_single

    def _get_json_track(self):
        json_track = []
        for key,value in self._info.items():
            # print(value['emerge'],self.output_absence,"GGGGGGGGGGGGgGGGGGGGgg")
            if value['emerge'] or self.output_absence:
                json_track.append( self._get_json_track_single(key, value) )
        return json_track

# if __name__ == "__main__":
#     import pprint
#     pp = pprint.PrettyPrinter(depth=6)

#     """
#     json_yolo = [{'confidences': [38], 'objectHeight': 248, 'objectPicY': 196,
#                   'objectTypes': ['bicycle'], 'objectWidth': 330, 'objectPicX': 229},"
#                 "{'confidences': [13], 'objectHeight': 41, 'objectPicY': 80,
#                   'objectTypes': ['person'], 'objectWidth': 38, 'objectPicX': 60},"
#                 "{'confidences': [82], 'objectHeight': 300, 'objectPicY': 218,
#                   'objectTypes': ['dog'], 'objectWidth': 257, 'objectPicX': 124},"
#                 "{'confidences': [14], 'objectHeight': 390, 'objectPicY': 97,
#                   'objectTypes': ['bicycle'], 'objectWidth': 485, 'objectPicX': 86},"
#                 "{'confidences': [73, 38], 'objectHeight': 89, 'objectPicY': 82,
#                   'objectTypes': ['car', 'truck'], 'objectWidth': 220, 'objectPicX': 466}]"
#     """

#     # data/dog.jpg
#     json_yolo = [{'confidences': [38], 'objectHeight': 248, 'objectPicY': 196, 'objectTypes': ['bicycle'], 'objectWidth': 330, 'objectPicX': 229}, {'confidences': [13], 'objectHeight': 41, 'objectPicY': 80, 'objectTypes': ['person'], 'objectWidth': 38, 'objectPicX': 60}, {'confidences': [82], 'objectHeight': 300, 'objectPicY': 218, 'objectTypes': ['dog'], 'objectWidth': 257, 'objectPicX': 124}, {'confidences': [14], 'objectHeight': 390, 'objectPicY': 97, 'objectTypes': ['bicycle'], 'objectWidth': 485, 'objectPicX': 86}, {'confidences': [73, 38], 'objectHeight': 89, 'objectPicY': 82, 'objectTypes': ['car', 'truck'], 'objectWidth': 220, 'objectPicX': 466}]
#     #pprint(json_yolo)

#     #tracker = Sort(height_max=576, width_max=768, max_history=100, categories_ignored='dog')
#     tracker = Sort(height_max=576,
#                    width_max=768,
#                    min_hits=0,
#                    max_miss=1,
#                    categories='dog', # default, [], is not to ingore anything
#                    record_absence=0,
#                    output_absence=0)

#     for n in range(9):
#         time0 = time.time()
#         json_track = tracker.tracking(json_yolo)
#         #json_track = tracker.tracking(( json_yolo  if n%3!=0 else  [] ))
#         #json_track = tracker.tracking(( json_yolo  if n%3==0 else  [] ))
#         duration = time.time()-time0
#         pp.pprint(json_track)
#         pp.pprint(tracker.death)
#         print('='*20)
#         #print(tracker.matches_id)
#         print(duration)

#     del tracker

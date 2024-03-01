'''
Prepare:
$ docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:management
$ docker start mssql
$ conda activate aloha
$ python yyds/sql_sq.py
$ python sql_s.py
$ python event_ss.py 
'''




import pika
import json
import time
class DetectSendor(object):
    def __init__(self, qName='event_queue', hostName='localhost'):
        self.qName = qName
        self.hostName = hostName

    def sendEvents(self,eventDicts):
        '''
        eventDicts : [{"start_time":current_time, "user_id":9487,
                       "uu_id":args.user_id,"event_action": args.event_action,
                       "status":args.status,"group_id":'G01',"location_id":1,
                       "confidence":args.confidence,"prediction_status":args.prediction,
                       "event_action_id_2nd": 2,"confidence_2nd":97,"event_action_id_3rd": 122,"confidence_3rd":97,
                       "snapshot":'/home/samba/raw_result/S1_stand_up_00.png',"center_x":2900,"center_y":100 }
        ,{},{},{}.........]
        Where : 
            status,confidence,user_id,prediction,center_x,center_y --> int
            uuid --> string
            current_time --> time.time() object
            snapshot --> string
        '''
        # print(eventDicts)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.hostName))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.qName)

        for eventDict in eventDicts:
            #eventDict = {'start_time': 1692085755.2212994, 'user_id': 9487, 'uu_id': '124', 'event_action': 122, 'status': 3, 'group_id': 'G01'      , 'location_id': 1, 'confidence': 72, 'prediction_status': 0, 'event_action_id_2nd': 2, 'confidence_2nd': 97, 'event_action_id_3rd': 122, 'confidence_3rd': 97, 'snapshot': '/home/samba/raw_result/S1_stand_up_00.png', 'center_x': 2900, 'center_y': 100}
            self.channel.basic_publish(exchange='', routing_key=self.qName, body=json.dumps(eventDict))
            time.sleep(0.0005)
        self.connection.close()
        return 0



class DetectSendor4Pred(object):
    def __init__(self, qName='pred_queue', hostName='localhost'):
        self.qName = qName
        self.hostName = hostName

    def sendEvents(self,eventDicts):
        '''
        eventDicts : [{"start_time":current_time, "user_id":9487,
                       "uu_id":args.user_id,"event_action": args.event_action,
                       "status":args.status,"group_id":'G01',"location_id":1,
                       "confidence":args.confidence,"prediction_status":args.prediction,
                       "event_action_id_2nd": 2,"confidence_2nd":97,"event_action_id_3rd": 122,"confidence_3rd":97,
                       "snapshot":'/home/samba/raw_result/S1_stand_up_00.png',"center_x":2900,"center_y":100 }
        ,{},{},{}.........]
        Where : 
            status,confidence,user_id,prediction,center_x,center_y --> int
            uuid --> string
            current_time --> time.time() object
            snapshot --> string
        '''
        # print(eventDicts)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.hostName))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.qName)

        for eventDict in eventDicts:
            #eventDict = {'start_time': 1692085755.2212994, 'user_id': 9487, 'uu_id': '124', 'event_action': 122, 'status': 3, 'group_id': 'G01'      , 'location_id': 1, 'confidence': 72, 'prediction_status': 0, 'event_action_id_2nd': 2, 'confidence_2nd': 97, 'event_action_id_3rd': 122, 'confidence_3rd': 97, 'snapshot': '/home/samba/raw_result/S1_stand_up_00.png', 'center_x': 2900, 'center_y': 100}
            self.channel.basic_publish(exchange='', routing_key=self.qName, body=json.dumps(eventDict))
            # time.sleep(0.005)
            time.sleep(0.0005)
        self.connection.close()
        return 0


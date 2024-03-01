#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import pika
import queue
import json
import time
import ast
import os
from enum import IntEnum
from datetime import datetime
import numpy as np
# from db.send_sq import SqlRpcClient
from modules.send_pred import SqlSimpleClient
# from alert.send_alert import AlertSimpleClient
import math
import requests
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))

event_list = []
exist_uuid = []
detection_server_ip = 'localhost'
detection_queue = "pred_queue"
event_server_ip = 'localhost'
event_queue = "event_queue"
# drop_id = []
EVENT_TIMEOUT = 10
LIST_LENGTH = 300
DO_PREDICTION = True


prediction_model = load_model("weights/train_model_3.h5")

# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '-s', '--status', type=int, default=3,
#         help='status of event action')
#     parser.add_argument(
#         '-p', '--prediction', type=int, default=0,
#         help='prediction of event action')
#     parser.add_argument(
#         '-e', '--event_action', type=int, default=1,
#         help='event action')    
#     parser.add_argument(
#         '-u', '--user_id', type=int, default=1,
#         help='user_id')    
#     parser.add_argument(
#         '-c', '--confidence', type=int, default=72,
#         help='confidence')  
#     args = parser.parse_args()
#     return args 


def detect_list_process(act_result):
    
    array = np.zeros(shape = (123,))
    for i in act_result:
        array[int(i[0])] = i[1]
    array = np.expand_dims(array,axis=0)
    
    return array
    
def predict_preprocess(action_list):

    assert len(action_list) == LIST_LENGTH, print(f'wrong')

    unit = int(LIST_LENGTH / 10)

    pred_list = []
    for i in range(10):
        arr_ = action_list[i*unit:(i+1)*unit]
        sum_ = sum(arr_) / np.sum(arr_)
        pred_list.append(sum_)
    pred_array = np.array(pred_list)
    pred_array = np.transpose(pred_array,(1,0,2))
    return pred_array



def do_predict(item_n):
    
    json = item_n.get("json")
    action_list = json.get("action_list")
    pred_array = predict_preprocess(action_list)
    action_list.pop(0)
    pred_re = prediction_model.predict(pred_array)
    pred_ind = np.argsort(np.sum(pred_re[0],axis=0))[-3:][::-1]
    pred_conf = []
    for ind in pred_ind:
        sum_ = np.sum(pred_re[0],axis=0)
        pred_conf.append(f"{int(sum_[ind]/np.sum(pred_re[0])*100)}")
    
    json.update({"prediction_status":int(1)})
    # json.update({"event_action":pred_ind[0]})
    json.update({"event_action":int(124)})
    json.update({"confidence":pred_conf[0]})
    json.update({"event_action_id_2nd":int(pred_ind[1])})
    json.update({"confidence_2nd":pred_conf[1]})
    json.update({"event_action_id_3rd":int(pred_ind[2])})
    json.update({"confidence_3rd":pred_conf[2]})

    json_n = json.copy()
    del json_n["action_list"]
    # del json_n["drop_id"]
    
    return json_n


    
def check_event_uuid(uuid):
    
    uuid_exist = False
    myjson = {'uuid_exist':uuid_exist}
    if uuid in exist_uuid:
        print( f"UUID {uuid} exists !!!!!")
        uuid_exist = True
        item = event_list[exist_uuid.index(uuid)]
        myjson = {'uuid_exist':uuid_exist, "json": item}
    return myjson

def check_drop_uuid(drop_id):

    if len(drop_id) > 0:
        print(f"DROP UUID {drop_id}")
        for rm_id in drop_id:
            ind = exist_uuid.index(rm_id)
            exist_uuid.remove(rm_id)
            event_list.pop(ind)
            
            print( f"UUID {rm_id} removed !!!!!")

def remove_uuid(uuid):
    
    ind = exist_uuid.index(uuid)
    exist_uuid.remove(uuid)
    event_list.pop(ind)
    print( f"UUID {uuid} removed !!!!!")
    

def action_append(json,new_action):
    global event_list
    uuid = json.get("uu_id")
    new_action = new_action
    action_list = json.get("action_list")
    action_list.append(new_action)
    json['action_list'] = action_list
    item = event_list[exist_uuid.index(uuid)]
    item['list_length'] += 1
    item['json'] = json
    event_list[exist_uuid.index(uuid)] = item
    return item
        
def process_content(queue, jd):
    global exist_uuid
    print(f"Processing: {jd}")
    icount = 0
    uuid = jd.get("uu_id")

    
    
    while True:
        if queue.empty():
            time.sleep(1)
            # print(f"{queue} is empty~~~~~~~~")
            icount += 1
            # break
        else:
            icount = 0
            msg = queue.get()
            cmd = msg.get('cmd')
            new_action = msg.get("new_action")
            # drop_id = msg.get("drop_id")
            
            # print(f"{uuid} $$$$$$$$$$$$$$$$$$")
            
            if cmd == "append":
                
                item_n = action_append(jd,new_action)
                # print(item_n)
                print(f"####### UUID {uuid} New action append #######")
                queue.put(item_n)

                list_length = item_n.get("list_length")
                print(f"LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL   {list_length}")
                # print(alert_t)
                if list_length >= LIST_LENGTH and DO_PREDICTION:
                    alert_pred = SqlSimpleClient()
                    print(f"UUID {uuid} DOOOOOOO PERDICTION!!!")
                    send_json = do_predict(item_n)
                    alert_pred.call(send_json)
                    print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEND")
                    
                    

            if cmd == "quit":
                
                print(f"{uuid} 收到 quit 訊息，即將結束")
                # check_drop_uuid(drop_id)
                # exist_uuid = list(set(exist_uuid) - set(drop_id))
                remove_uuid(uuid)
                break
        
        if icount >= EVENT_TIMEOUT:

            item = event_list[exist_uuid.index(uuid)]
            print(f"{uuid} change status to 881 !!!!!!!!!!")
            item.update({"status":881})
            icount = 0

        print(f"************收到訊息 {uuid} {icount}*****************")
                
    
    
def process_message(channel, method, properties, body):
    
    jd = json.loads(body)
    action_list = []
    uuid = jd.get("uu_id")
    print(uuid)
    action_result = jd.get("actResults")
    new_action = detect_list_process(action_result) # cvt list to array
    action_list.append(new_action)
    # drop_id = jd['drop_id']
    id_check = check_event_uuid(uuid)
    uuid_exist = id_check.get("uuid_exist")
    
    # if len(drop_id) > 0:
    #     item = id_check.get("json")
    #     myjson = {'cmd':"quit", "drop_id":drop_id}
    #     thread_queue = item.get('queue_id')
    #     thread_queue.put(myjson)   
    
    if uuid_exist:
        item = id_check.get("json")
        icount = item.get("icount")
        myjson = {'cmd':"append", "new_action":new_action, "icount":icount}
        
        thread_queue = item.get('queue_id')
        thread_queue.put(myjson)


    else:
    
        print( f"New UUID and create new thread !!!!!")
        exist_uuid.append(uuid)
        # print(exist_uuid)
        q = queue.Queue()
        t = threading.Thread(target=process_content, args=(q, jd,))
        t.setDaemon(True)
        jd.pop('actResults')
        jd['action_list'] = action_list
        new_item = { "json": jd, "thr_id": t, "queue_id": q, "list_length": 1,
                     "icount": 1, "status": 1}
        # print(uuid,q)
        event_list.append(new_item)
        t.start()

def alert_threading(alert_queue, event_queue, channel_e):
    # ttl = 5000
    # properties = pika.BasicProperties(expiration=str(ttl))
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    print(alert_queue)
    while True:
        # time.sleep(0.3)
        if alert_queue.empty():
            time.sleep(1)
            # print("#%#$#$%#$%#$%#$%#$%#$#$%#$#%$^^%^%$%$%^$%$%$%$%$^%$^$^%$%^$%^$^%$^%$^%^")
        else:
            myjson = alert_queue.get()
            ret = channel_e.basic_publish(exchange='', routing_key=event_queue, body=json.dumps(myjson),properties=properties)
            print('PRDIECTED JSON SEND !!!!#%#$#$%#$%#$%#$%#$%#$#$%#$#%$^^%^%$%$%^$%$%$%$%$^%$^$^%$%^$%^$^%$^%$^%^')
            

    
        
def monitor_thread():
    count = 0
    while True:
        time.sleep(1)
        count += 1
        for x in event_list:
            status = int(x.get("status"))
            if status == 881:
                print("Prepare to terminate thread")
                qid = x.get("queue_id")
                myjson = { "cmd": "quit","item":x}
                qid.put(myjson)
            jd = x.get("json")
            uuid = jd.get("uuid")
            thr_id = x.get("thr_id")
            que_id = x.get("queue_id")
            if (not thr_id.is_alive()):
                print(f"**************************************{uuid} <<=uuid  {thr_id}<<==thread_id  {que_id}<<==queue_id*************************************************")
                if (count % 5) == 1:
                    remove_uuid(uuid)
                    
    # print("********exit thread*********")
        # if len(drop_id) > 0:
        #     print("MMMMMMMMMMMMMMMMMMMMMMMMMMM")
        #     for rem_id in drop_id:
        #         print(f"Will Remove UUID {rem_id}")
        #         ind = exist_uuid.index(rem_id)
        #         item = event_list[ind]
        #         quit_queue = item.get('queue_id')
        #         myjson = {"cmd": "quit", "rem_id": remid}
        #         quit_queue.put(myjson)
        # # else:
        #     print("NNNNNNNNNNNNNNNNNNNNNN")
            
        
        
        
# global alert_queue, alert_t
# alert_queue = queue.Queue()
# print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
# print(alert_queue)

# connection = pika.BlockingConnection(pika.ConnectionParameters(event_server_ip))
# channel_e = connection.channel()
# channel_e.queue_declare(queue=event_queue)

# alert_t = threading.Thread(target=alert_threading,args=(alert_queue, event_queue,channel_e))
# alert_t.setDaemon(True)
# alert_t.start()


            
    
monitor_t = threading.Thread(target=monitor_thread)
monitor_t.setDaemon(True)

monitor_t.start()

connection = pika.BlockingConnection(pika.ConnectionParameters(detection_server_ip))

channel = connection.channel()

channel.queue_declare(queue = detection_queue)

channel.basic_consume(queue= detection_queue, on_message_callback=process_message, auto_ack=True)
print("waiting message........")



try:
    # Enter a blocking loop to consume messages
    channel.start_consuming()
except KeyboardInterrupt:
    # Gracefully stop the connection on CTRL+C
    channel.stop_consuming()

# Close the connection
connection.close()

    
    
    
    




    





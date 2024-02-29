#!/usr/bin/env python
import pika
import json
###########################################
###  change event_action type to trigger event new count
###########################################
class SqlSimpleClient(object):
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='event_queue')

    def call(self, myjson):
        self.response = None
        self.channel.basic_publish(exchange='', routing_key='event_queue', body=json.dumps(myjson))
        self.connection.close()
        return self.response
    


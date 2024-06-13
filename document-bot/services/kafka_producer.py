import configs.env as env
from confluent_kafka import Producer

def create_producer():
    conf = {
        'bootstrap.servers': env.KAFKA_BROKER
    }

    producer = Producer(conf)

    return producer
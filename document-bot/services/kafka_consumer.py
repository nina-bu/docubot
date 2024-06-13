import sys
from confluent_kafka import Consumer, KafkaError, KafkaException
import configs.env as env
import logging
from services import event_service
from routers import documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_consumer(topic: str):
    conf = {
        'bootstrap.servers': env.KAFKA_BROKER,
        'group.id': 'orchestrator-group',
        'auto.offset.reset': 'latest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    return consumer


def consume_messages(consumer):
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None: continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f'{msg.topic()} [{msg.partition()}] reached end at offset {msg.offset()}')
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                event = event_service.parse_event(msg.value().decode("utf-8"))
                # print(event)
                documents.saga_create(event)
                logger.info(f'Received message: {msg.value().decode("utf-8")}')
    finally:
        consumer.close()

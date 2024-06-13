from threading import Thread
import uvicorn
from fastapi import FastAPI
from routers import documents, lectures, projects, reports
from services.kafka_consumer import create_consumer, consume_messages

app = FastAPI()

app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(lectures.router)
app.include_router(reports.router)

def start_kafka_consumer():
    consumer = create_consumer('document-bot-success')
    consume_messages(consumer)


if __name__ == '__main__':
    kafka_thread = Thread(target=start_kafka_consumer, daemon=True)
    kafka_thread.start()
    uvicorn.run(app, host="127.0.0.1", port=8000)

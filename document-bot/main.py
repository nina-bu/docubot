import uvicorn
from fastapi import FastAPI
from py_eureka_client import eureka_client
from routers import documents, lectures, projects

app = FastAPI()

app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(lectures.router)

def init_eureka_client():
    eureka_client.init_registry_client(
        eureka_server="http://eureka-server:8761/eureka",
        app_name="docubot-service", 
        instance_host="localhost",
        instance_port=8000, 
        instance_health_check_url="/test-milvus-connection/",
        use_dns=False
    )


if __name__ == '__main__':
    # init_eureka_client()
    uvicorn.run(app, host="127.0.0.1", port=8000)

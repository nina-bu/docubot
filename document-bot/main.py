import uvicorn

from fastapi import FastAPI
from routers import projects

app = FastAPI()

app.include_router(projects.router)

# Initialize Eureka client (uncomment if needed)
# init_eureka_client()

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

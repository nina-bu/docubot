import uvicorn
from fastapi import FastAPI
from routers import documents, projects, lectures

app = FastAPI()

app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(lectures.router)

# Initialize Eureka client (uncomment if needed)
# init_eureka_client()

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

import uvicorn
from fastapi import FastAPI
from routers import documents, lectures, projects, reports

app = FastAPI()

app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(lectures.router)
app.include_router(reports.router)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

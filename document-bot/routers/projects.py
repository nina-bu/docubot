import configs.env as env
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.embedding_service import embed
from services.milvus_service import client

router = APIRouter()

class VectorCreateRequest(BaseModel):
    project_id: int
    name: str
    description: str
    type: str

class VectorUpdateRequest(BaseModel):
    name: str
    description: str
    type: str

class Vector(BaseModel):
    project_id: int
    name: str
    description: str
    type: str
    name_emb: list
    descr_emb: list

@router.post("/api/v1/collections/{collection_name}")
async def create(collection_name: str, project: VectorCreateRequest):
    try:
        vector = Vector(
            project_id=project.project_id,
            name=project.name,
            description=project.description,
            type=project.type,
            name_emb=embed(project.name),
            descr_emb=embed(project.description)
        )
        client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={"message": f"Vector has been successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/api/v1/collections/{collection_name}/{vector_id}")
async def delete(collection_name: str, vector_id: int):
    try:
        vector_data = client.get(collection_name=collection_name, ids=vector_id)
        if vector_data:
            vector_dict = {
                "project_id": vector_data[0]["project_id"],
                "name": vector_data[0]["name"],
                "description": vector_data[0]["description"],
                "type": vector_data[0]["type"],
            }       
            return JSONResponse(content={"vector_data": vector_dict})
        else:
            return JSONResponse(content={"message": "Vector not found"}, status_code=404)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   
    
@router.put("/api/v1/collections/{collection_name}/{vector_id}")
async def update(collection_name: str, vector_id: int, project: VectorUpdateRequest):
    try:
        vector_data = Vector(
            project_id=vector_id,
            name=project.name,
            description=project.description,
            type=project.type,
            name_emb=embed(project.name),
            descr_emb=embed(project.description)
        )
        client.upsert(collection_name=collection_name, vector_id=vector_id, data=vector_data.dict())
        return JSONResponse(content={"message": f"Vector with ID {vector_id} has been successfully updated."}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.delete("/api/v1/collections/{collection_name}/{vector_id}")
async def delete(collection_name: str, vector_id: int):
    try:
        client.delete(collection_name=collection_name, pks=vector_id)
        return JSONResponse(content={"message": f"Vector with ID {vector_id} has been successfully deleted."}, status_code=200)
   
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
from configs.env import PROJECT_COLLECTION_NAME
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.embedding_service import embed
from services.milvus_service import client

router = APIRouter()

class VectorCreateRequest(BaseModel):
    id: int
    name: str
    description: str
    type: str

class VectorUpdateRequest(BaseModel):
    name: str
    description: str
    type: str

class VectorResponse(BaseModel):
    id: int
    name: str
    description: str
    type: str

class Vector(BaseModel):
    id: int
    name: str
    description: str
    type: str
    name_emb: list
    descr_emb: list

collection_name = PROJECT_COLLECTION_NAME

@router.post("/api/v1/collections/projects")
async def create(project: VectorCreateRequest):
    try:
        vector = Vector(
            id=project.id,
            name=project.name,
            description=project.description,
            type=project.type,
            name_emb=embed(project.name),
            descr_emb=embed(project.description)
        )
        client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={"message": f"Vector successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/api/v1/collections/projects/{vector_id}", response_model=VectorResponse)
async def get(vector_id: int):
    try:
        vector_data = client.get(collection_name=collection_name, ids=vector_id)
        if vector_data:
            vector = vector_data[0]
            return VectorResponse(
                id=vector["id"],
                name=vector["name"],
                description=vector["description"],
                type=vector["type"]
            )
        else:
            return JSONResponse(content={"message": "Vector not found"}, status_code=404)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   
    
@router.put("/api/v1/collections/projects/{vector_id}")
async def update(vector_id: int, project: VectorUpdateRequest):
    try:
        vector_data = Vector(
            id=vector_id,
            name=project.name,
            description=project.description,
            type=project.type,
            name_emb=embed(project.name),
            descr_emb=embed(project.description)
        )
        client.upsert(collection_name=collection_name, vector_id=vector_id, data=vector_data.dict())
        return JSONResponse(content={"message": f"Vector with ID {vector_id} successfully updated."}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.delete("/api/v1/collections/projects/{vector_id}")
async def delete(vector_id: int):
    try:
        client.delete(collection_name=collection_name, pks=vector_id)
        return JSONResponse(content={"message": f"Vector with ID {vector_id} successfully deleted."}, status_code=200)
   
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
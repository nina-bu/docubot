from configs.env import DOCUMENT_COLLECTION_NAME
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.embedding_service import chunk, embed_insert
from services.milvus_service import client

router = APIRouter()

class VectorCreateRequest(BaseModel):
    project_id: int
    document_id: int
    name: str
    version: str
    text: str

class VectorResponse(BaseModel):
    id: int
    project_id: int
    document_id: int
    name: str
    version: str
    chunk_id: int
    text: str


class Vector(BaseModel):
    project_id: int
    document_id: int
    name: str
    version: int
    text: str
    chunk_id: int
    txt_emb: list

collection_name = DOCUMENT_COLLECTION_NAME

# FEAT: CRUD
@router.post("/api/v1/collections/documents")
async def create(document: VectorCreateRequest):
    try:
        for chunk_id, chunk_txt in chunk(document.text):
            vector = Vector(
                project_id=document.project_id,
                document_id=document.document_id,
                name=document.name,
                version=document.version,
                text=chunk_txt,
                chunk_id=chunk_id, 
                txt_emb=embed_insert(chunk_txt)
            )
            client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={"message": f"Vector(s) successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# FEAT: SIMPLE QUERY 1 - Retrieval by ID
@router.get("/api/v1/collections/documents/{vector_id}", response_model=VectorResponse)
async def get(vector_id: int):
    try:
        vector_data = client.get(collection_name=collection_name, ids=vector_id)
        if vector_data:
            vector = vector_data[0]
            return VectorResponse(
                id=vector["id"],
                project_id=vector["project_id"],
                document_id=vector["document_id"],
                name=vector["name"],
                version=vector["version"],
                chunk_id=vector["chunk_id"],
                text=vector["text"]
            )
        else:
            return JSONResponse(content={"message": "Vector not found"}, status_code=404)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   
    
@router.delete("/api/v1/collections/documents/{vector_id}")
async def delete(vector_id: int):
    try:
        client.delete(collection_name=collection_name, pks=vector_id)
        return JSONResponse(content={"message": f"Vector with ID {vector_id} successfully deleted."}, status_code=200)
   
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# TODO get by project, document and version with LLM 
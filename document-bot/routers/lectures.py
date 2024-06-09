from typing import Optional
from fastapi import APIRouter
from configs.env import LECTURES_COLLECTION_NAME
from services.embedding_service import chunk, embed
from pydantic import BaseModel
from services.milvus_service import client
from fastapi.responses import JSONResponse

class LectureCreateRequest(BaseModel):
    name: str
    content: str
    difficulty: int
    min_recommended_age: int
    max_recommended_age: int
    creator_id: int

class LectureVector(BaseModel):
    name: str
    content: str
    difficulty: int
    min_recommended_age: int
    max_recommended_age: int
    creator_id: int
    chunk_id: int
    name_emb: list
    content_emb: list

class LectureGetResponse(BaseModel):
    id: int
    name: str
    content: str
    difficulty: int
    min_recommended_age: int
    max_recommended_age: int
    creator_id: int
    chunk_id: int


router = APIRouter()

collection_name = LECTURES_COLLECTION_NAME

@router.post("/api/v1/collections/lectures")
async def create(lecture: LectureCreateRequest):
    try:
        for chunk_id, chunk_txt in chunk(lecture.content):
            vector = LectureVector(
                name=lecture.name,
                content=chunk_txt,
                difficulty=lecture.difficulty,
                min_recommended_age=lecture.min_recommended_age,
                max_recommended_age=lecture.max_recommended_age,
                creator_id=lecture.creator_id,
                chunk_id=chunk_id,
                name_emb=embed(lecture.name),
                content_emb=embed(chunk_txt)
            )
            client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={'message': f"Vector(s) successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
    

@router.get("/api/v1/collections/lectures/{id}", response_model=LectureGetResponse)
async def get(id: int):
    try:
        vector_data = client.get(collection_name=collection_name, ids=id)
        if vector_data:
            vector = vector_data[0]
            return LectureGetResponse(
                id=vector["id"],
                name=vector["name"],
                content=vector["content"],
                difficulty=vector["difficulty"],
                min_recommended_age=vector["min_recommended_age"],
                max_recommended_age=vector["max_recommended_age"],
                creator_id=vector["creator_id"],
                chunk_id=vector["chunk_id"]
            )
        else:
            return JSONResponse(content={'message': 'Vector not found'},
                                status_code=404)
        
    except Exception as e:
         return JSONResponse(content={"error": str(e)}, status_code=500)
    
@router.put("/api/v1/collections/lectures/{id}")
async def update(id: int, lecture: LectureCreateRequest):
    existing_entity = client.get(collection_name=collection_name, ids=id)
    if existing_entity:
        client.delete(collection_name=collection_name, pks=id)

    try:
        for chunk_id, chunk_txt in chunk(lecture.content):
            vector = LectureVector(
                name=lecture.name,
                content=chunk_txt,
                difficulty=lecture.difficulty,
                min_recommended_age=lecture.min_recommended_age,
                max_recommended_age=lecture.max_recommended_age,
                creator_id=lecture.creator_id,
                chunk_id=chunk_id,
                name_emb=embed(lecture.name),
                content_emb=embed(chunk_txt)
            )
            client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={'message': f"Vector(s) successfully updated."}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
    
@router.delete("/api/v1/collections/lectures/{id}")
async def delete(id: int):
    try:
        client.delete(collection_name=collection_name, pks=id)
        return JSONResponse(content={"message": f"Vector with ID {id} successfully deleted."}, status_code=200)
   
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
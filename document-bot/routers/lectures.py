from typing import Optional
from fastapi import APIRouter
from fastapi.params import Query
from pymilvus import AnnSearchRequest, Collection, WeightedRanker
from configs.env import LECTURES_COLLECTION_NAME
from services.embedding_service import chunk, embed_search, embed_insert
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
collection = Collection(name=collection_name)

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
                name_emb=embed_insert(lecture.name),
                content_emb=embed_insert(chunk_txt)
            )
            client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={'message': f"Vector(s) successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
    
# FEAT: SIMPLE QUERY 1 - Retreival by ID
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
                name_emb=embed_insert(lecture.name),
                content_emb=embed_insert(chunk_txt)
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
    
# FEAT: SIMPLE QUERY 2 - Single vector search
@router.get("/api/v1/collections/lecturess/vector-search")
async def vector_search(search_text: str = Query(..., description="The text to search for")):
    try:
        vector_data = single_vector_search(search_text)
        if vector_data:
            return vector_data
        else:
            return JSONResponse(content={"message": "No vectors match the search."}, status_code=204)
        
    except Exception as e:
         return JSONResponse(content={"error": str(e)}, status_code=500)
    
# FEAT: COMPLEX QUERY 1 - Hybrid Search (name and content)
@router.get("/api/v1/collections/lecturess/hybrid-search")
async def hybrid_search(name_search_text: str = Query(..., description="The text to search for"), content_search_text: str = Query(..., description="The text to search for")):
    try:
        vector_data = multiple_vector_ann_search(name_search_text, content_search_text)
        if vector_data:
            return vector_data
        else:
            return JSONResponse(content={"message": "No vectors match the search."}, status_code=204)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

# FEAT: COMPLEX QUERY 2 - Single vector search with filters (difficulty and creator_id)
@router.get("/api/v1/collections/lecturess/filter")
async def vector_search_with_filters(difficulty: int = Query(..., gt=0, lt=3), creator_id: int = Query(...), content_search_text: str = Query(..., description="The text to search for")):
    try:
        vector_data = single_vector_search_with_filters(content_search_text, f'difficulty == {difficulty} and creator_id == {creator_id}')
        if vector_data:
            return vector_data
        else:
            return JSONResponse(content={"message": "No vectors match the search."}, status_code=204)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# FEAT: COMPLEX QUERY 3 - Single vector search with filters (age range)
@router.get("/api/v1/collections/lecturess/filter_age")
async def vector_search_with_filters_age(age: int = Query(...), content_search_text: str = Query(..., description="The text to search for")):
    try:
        vector_data = single_vector_search_with_filters(content_search_text, f'min_recommended_age <= {age} and max_recommended_age >= {age}')
        if vector_data:
            return vector_data
        else:
            return JSONResponse(content={"message": "No vectors match the search."}, status_code=204)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def single_vector_search(search_term: str):
    search_vector = embed_search(search_term)

    return client.search(
        collection_name=collection_name,
        data=search_vector,
        anns_field="content_emb",
        output_fields=['name', 'content', 'difficulty', 'min_recommended_age', 'max_recommended_age', 'creator_id'],
        limit=10
    )

def multiple_vector_ann_search(name_search_term: str, content_search_term: str):
    name_search_vector = embed_search(name_search_term)

    search_param_1 = {
        "data":name_search_vector,
        "anns_field":"name_emb",
        "param":{
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }, 
        "limit":5
    }      
    request_1 = AnnSearchRequest(**search_param_1)

    content_search_vector = embed_search(content_search_term)

    search_param_2 = {
        "data":content_search_vector,
        "anns_field":"content_emb",
        "param":{
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }, 
        "limit":5
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]
    rerank = WeightedRanker(0.4, 0.6) 

    collection.load()
    res = collection.hybrid_search(
        reqs, 
        rerank,
        limit=5, 
        output_fields=['name', 'content', 'difficulty', 'min_recommended_age', 'max_recommended_age', 'creator_id']
    )
    return res

def single_vector_search_with_filters(search_term: str, filter: str):
    search_vector = embed_search(search_term)

    return client.search(
        collection_name=collection_name,
        data=search_vector,
        anns_field="content_emb",
        output_fields=['name', 'content', 'difficulty', 'min_recommended_age', 'max_recommended_age', 'creator_id'],
        filter=filter,
        limit=10
    )
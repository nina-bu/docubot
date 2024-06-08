from configs.env import MILVUS_HOST, MILVUS_PORT, PROJECT_COLLECTION_NAME
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymilvus import AnnSearchRequest, Collection, WeightedRanker, connections
from services.embedding_service import embed_insert, embed_search
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

class VectorSearchRequest(BaseModel):
    search_term: str

class Vector(BaseModel):
    id: int
    name: str
    description: str
    type: str
    name_emb: list
    descr_emb: list

collection_name = PROJECT_COLLECTION_NAME
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(name=PROJECT_COLLECTION_NAME)

# FEAT: CRUD
@router.post("/api/v1/collections/projects")
async def create(project: VectorCreateRequest):
    try:
        vector = Vector(
            id=project.id,
            name=project.name,
            description=project.description,
            type=project.type,
            name_emb=embed_insert(project.name),
            descr_emb=embed_insert(project.description)
        )
        client.insert(collection_name=collection_name, data=vector.dict())
        return JSONResponse(content={"message": f"Vector successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# FEAT: SIMPLE QUERY 1 - Retrieval by ID
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
            name_emb=embed_insert(project.name),
            descr_emb=embed_insert(project.description)
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

# FEAT: COMPLEX QUERY 1 - Hybrid Search
@router.post("/api/v1/collections/projects/search")
async def search(search: VectorSearchRequest):
    try:
        vector_data = hybrid_search(search.search_term)
        if vector_data:
            return vector_data
        else:
            return JSONResponse(content={"message": "No vectors match the search."}, status_code=204)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   


def hybrid_search(search_term: str):
    searh_vector = embed_search(search_term)

    search_param_1 = {
        "data":searh_vector,
        "anns_field":"name_emb",
        "param":{
            "metric_type": "L2",
            "params": {"nprobe": 12}
        }, 
        "limit":5
    }      
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data":searh_vector,
        "anns_field":"descr_emb",
        "param":{
            "metric_type": "L2",
            "params": {"nprobe": 12}
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
        limit=2, 
        output_fields=['name', 'description', 'type']
    )
    return res

from configs.env import PROJECT_COLLECTION_NAME
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
    budget: int
    type: str

class VectorUpdateRequest(BaseModel):
    name: str
    description: str
    budget: int
    type: str

class VectorResponse(BaseModel):
    id: int
    name: str
    description: str
    budget: int
    type: str

class VectorSearchRequest(BaseModel):
    search_term: str

class VectorFilterRequest(BaseModel):
    description: str
    lower_budget: int
    upper_budget: int
    type: str

class VectorIterateRequest(BaseModel):
    name: str
    budget: int
    
class Vector(BaseModel):
    id: int
    name: str
    description: str
    budget: int
    type: str
    name_emb: list
    descr_emb: list

collection_name = PROJECT_COLLECTION_NAME
collection = Collection(name=PROJECT_COLLECTION_NAME)

# FEAT: CRUD
@router.post("/api/v1/collections/projects")
async def create(project: VectorCreateRequest):
    try:
        vector = Vector(
            id=project.id,
            name=project.name,
            description=project.description,
            budget=project.budget,
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
                budget=vector["budget"],
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
            budget=project.budget,
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

# FEAT: COMPLEX QUERY 1 - Hybrid Search (name and description)
@router.post("/api/v1/collections/projects/search")
async def search(search_req: VectorSearchRequest):
    try:
        return hybrid_search(search_req.search_term)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   

# FEAT: COMPLEX QUERY 2 - Vector Search with filters (budget and type)
@router.post("/api/v1/collections/projects/filter")
async def filter(filter_req: VectorFilterRequest):
    try:
        return filter_search(filter_req.description, 
                                    filter_req.lower_budget, 
                                    filter_req.upper_budget, 
                                    filter_req.type)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)  
     
# FEAT: COMPLEX QUERY 4 - Iterated Vector Search with filters 
@router.post("/api/v1/collections/projects/iterate")
async def iterate(iterate_req: VectorIterateRequest):
    try:
        return iterator_filter(iterate_req.name, iterate_req.budget)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) 
    
def hybrid_search(search_term: str):
    searh_vector = embed_search(search_term)

    search_param_1 = {
        "data":searh_vector,
        "anns_field":"name_emb",
        "param":{
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }, 
        "limit":5
    }      
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data":searh_vector,
        "anns_field":"descr_emb",
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
        limit=2, 
        output_fields=['name', 'description', 'budget', 'type']
    )
    return res

def filter_search(description, lower_budget, upper_budget, type):
    descr_vector = embed_search(description)

    return client.search(
        collection_name=collection_name,
        data=descr_vector,
        anns_field="descr_emb",
        filter=f'{lower_budget} <= budget <= {upper_budget} and type == "{type}"',
        output_fields=['name', 'description', 'budget', 'type'],
        limit=10
    )

def iterator_filter(name, budget):    
    name_vector = embed_search(name)

    collection.load()
    iterator = collection.query_iterator(
        data=name_vector,
        anns_field="name_emb",    
        batch_size=5,
        limit=15, 
        expr=f'budget < {budget} and type == "INTERNAL"',
        output_fields=["name", "description", "type"]
    )   

    results = []
    while True:
        result = iterator.next()

        if len(result) == 0:
            iterator.close()
            break;
        
        for x in range(len(result)):
            results.append(result[x])

    return results

import csv
from fastapi import FastAPI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import Query
from fastapi.responses import JSONResponse
from py_eureka_client import eureka_client
from typing import List, Dict
import logging
import uvicorn

app = FastAPI()

def init_eureka_client():
    eureka_client.init(
        eureka_server="http://eureka-server:8761/eureka",
        app_name="vector-database-service",  # Name of your FastAPI service
        instance_host="localhost",
        instance_port=8000,  # Port where your FastAPI service is running
        instance_health_check_url="/test-milvus-connection/",  # Health check URL to ensure your service is healthy
        use_dns=False  # Set this to True if you want to use DNS for service discovery
    )

COLLECTION_NAME = 'onlineshop'
DIMENSION = 384
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
BATCH_SIZE = 128
TOP_K = 5
COUNT = 10000

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(name=COLLECTION_NAME)

# Check connection to Milvus and count items inserted into collection
@app.get("/test-milvus-connection/")
async def test_milvus_connection():
    try:
        # Check if connected to Milvus
        status = client.get_collection_stats(collection_name="onlineshop")
        return {"message": "Connected to Milvus", "status": status}
    except Exception as e:
        return {"message": "Error occurred during Milvus connection:", "error": str(e)}

# Count collection items
@app.get("/get-entity-count/")
async def get_entity_count():
    try:
        count = collection.num_entities
        return {"message": "Count of entities in collection", "count": count}
    except Exception as e:
        return {"message": "Error occurred while getting entity count:", "error": str(e)}

# Load SentenceTransformer model
transformer = SentenceTransformer('all-MiniLM-L6-v2')
client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

@app.get("/api/v1/collections/{collection_name}/getvector1/{vector_id}")
async def get_vector(
    collection_name: str,
    vector_id: int
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        vector_data = client.get(collection_name=collection_name, ids=[vector_id])

        if vector_data:
            vector_dict = {
                "id": vector_id,
                "description": vector_data[0]["description"],
                "category": vector_data[0]["category"],
                "location": vector_data[0]["location"]
            }
            return JSONResponse(content={"vector_data": vector_dict})
        else:
            return JSONResponse(content={"message": "Vector not found"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/v1/collections/{collection_name}/query")
async def query_collection(collection_name: str, filter: str = Query('Ne%', title="Filter", description="Filter query using LIKE operator"), limit: int = Query(10, title="Limit", description="Number of results to return")):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        res = client.query(
            collection_name=collection_name,
            filter=f"description LIKE '{filter}'",
            output_fields=["category", "description"],
            limit=limit
        )

        return {"query_result": res}
    except Exception as e:
        return {"error": str(e)}

model = SentenceTransformer('all-MiniLM-L6-v2')

def search_with_embedding(search_term: str) -> List[Dict[str, str]]:
    try:
        logging.basicConfig(level=logging.INFO)
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)

        search_params = {
            "metric_type": "L2"
        }

        results = client.search(
            collection_name='onlineshop',
            data=embedding,
            anns_field="descr_emb",
            search_params=search_params, 
            limit=20,
            output_fields=['description', 'location']
        )
   
        search_results = []
        for hit in results[0]:
            description = hit.get('description')
            location = hit.get('location')
            search_results.append({"description": description, "location": location})
        
        return search_results
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return [{"error": str(e)}]

@app.get("/api/v1/search/")
async def perform_search(search_term: str = Query(..., title="Search Term", description="Term to search for")):
    search_results = search_with_embedding(search_term)
    print("Received search term:", search_term)
    print("Search results:", search_results)

    return {"search_results": search_results}

# init_eureka_client()
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

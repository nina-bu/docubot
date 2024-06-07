import configs.env as env
from pymilvus import connections, Collection, MilvusClient

def connect_to_milvus():
    connections.connect(host=env.MILVUS_HOST, port=env.MILVUS_PORT)

def get_collection(collection_name: str):
    return Collection(name=collection_name)

client = MilvusClient(uri=f"http://{env.MILVUS_HOST}:{env.MILVUS_PORT}", token="root:Milvus")

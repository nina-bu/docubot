import configs.env as env
from pymilvus import Collection, MilvusClient, connections


def connect_to_milvus():
    connections.connect(host=env.MILVUS_HOST, port=env.MILVUS_PORT)

def create_milvus_client():
    client = MilvusClient(uri=f"http://{env.MILVUS_HOST}:{env.MILVUS_PORT}", token="root:Milvus")
    return client

def get_collection(collection_name: str):
    return Collection(name=collection_name)

connect_to_milvus()
client = create_milvus_client()

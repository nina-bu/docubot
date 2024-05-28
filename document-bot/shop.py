import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = 'onlineshop'  
DIMENSION = 384  
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = '19530'
BATCH_SIZE = 128
TOP_K = 3
COUNT = 10000

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
print('Connected to Milvus')

from pymilvus import utility
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='location', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='category', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='descr_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),  
    FieldSchema(name='ctg_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)      
]

schema = CollectionSchema(fields=fields)

collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.create_index(field_name="descr_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
collection.create_index(field_name="ctg_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
collection.load()
print('Collection created and indices created')


transformer = SentenceTransformer('all-MiniLM-L6-v2')

def csv_load(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row
        for row in reader:
            if '' in (row[0], row[1], row[2]):
                continue
            yield (row[0], row[1], row[2])
    
def embed_insert(data):
    embeds_descr = transformer.encode(data[1]) 
    embeds_ctg = transformer.encode(data[2])    
    ins = [
       
        data[0], data[1], data[2], 
        [x for x in embeds_descr],
        [x for x in embeds_ctg]
    ]
    collection.insert(ins)


count=0
data_batch = [[],[],[]]

try:
    for col0, col1, col2 in csv_load("data/shop_data.csv"):
        if count <= COUNT:
            data_batch[0].append(col0)
            data_batch[1].append(col1)
            data_batch[2].append(col2)
            
            if len(data_batch[0]) % BATCH_SIZE == 0:
                embed_insert(data_batch)
                data_batch = [[],[],[]]
            count += 1
            
        else:
            break

    if len(data_batch[0]) != 0:
        embed_insert(data_batch)
        count += len(data_batch[0])

    collection.flush()
    print('Inserted data successfully')
    print('Number of inserted items:', count)
except Exception as e:
    print('Error occurred during data insertion:', str(e))
    raise e  

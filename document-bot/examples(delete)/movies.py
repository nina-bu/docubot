COLLECTION_NAME = 'movies_db'  # Collection name
DIMENSION = 384  # Embeddings size
COUNT = 1000  # Number of vectors to insert
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

BATCH_SIZE = 128

TOP_K = 3

from pymilvus import connections

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

from pymilvus import utility

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

from pymilvus import FieldSchema, CollectionSchema, DataType, Collection


fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=200),  # VARCHARS need a maximum length, so for this example they are set to 200 characters
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
    'metric_type':'L2',
    'index_type':"IVF_FLAT",
    'params':{'nlist': 1536}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

import csv
from sentence_transformers import SentenceTransformer

transformer = SentenceTransformer('all-MiniLM-L6-v2')

def csv_load(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if '' in (row[1], row[7]):
                continue
            yield (row[1], row[7])


def embed_insert(data):
    embeds = transformer.encode(data[1]) 
    ins = [
            data[0],
            [x for x in embeds]
    ]
    collection.insert(ins)

import time

data_batch = [[],[]]

count = 0

for title, plot in csv_load("data/movie_plots.csv"):
    if count <= COUNT:
        data_batch[0].append(title)
        data_batch[1].append(plot)
        if len(data_batch[0]) % BATCH_SIZE == 0:
            embed_insert(data_batch)
            data_batch = [[],[]]
        count += 1
    else:
        break

if len(data_batch[0]) != 0:
    embed_insert(data_batch)

collection.flush()

search_terms = ['A movie about cars', 'A movie about monsters']

def embed_search(data):
    embeds = transformer.encode(data) 
    return [x for x in embeds]

search_data = embed_search(search_terms)

start = time.time()
res = collection.search(
    data=search_data,  # Embeded search value
    anns_field="embedding",  # Search across embeddings
    param={},
    limit = TOP_K,  # Limit to top_k results per search
    output_fields=['title']  # Include title field in result
)
end = time.time()

for hits_i, hits in enumerate(res):
    print('Title:', search_terms[hits_i])
    print('Search Time:', end-start)
    print('Results:')
    for hit in hits:
        print( hit.entity.get('title'), '----', hit.distance)
    print()

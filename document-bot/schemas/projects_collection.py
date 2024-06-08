import csv

import configs.env as env
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from services.embedding_service import transformer


def connect_to_milvus():
    connections.connect(host=env.MILVUS_HOST, port=env.MILVUS_PORT)
    print('Connected to Milvus')

def drop_collection_if_exists(collection_name):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

def create_collection_schema():
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name='budget', dtype=DataType.INT32),
        FieldSchema(name='type', dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name='name_emb', dtype=DataType.FLOAT_VECTOR, dim=env.DIMENSION),
        FieldSchema(name='descr_emb', dtype=DataType.FLOAT_VECTOR, dim=env.DIMENSION)
    ]
    return CollectionSchema(fields=fields)

def create_collection(collection_name, schema):
    collection = Collection(name=collection_name, schema=schema)

    collection.create_index(field_name="name_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
    collection.create_index(field_name="descr_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
    collection.load()

    print('Collection created and indices created')
    return collection

def csv_load(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for row in reader:
            if '' in (row[0], row[1], row[2], row[3], row[4]):
                continue
            yield (row[0], row[1], row[2], row[3], row[4])

def embed_insert(collection, data_batch, transformer):
    name_embeds = transformer.encode(data_batch[1])
    descr_embeds = transformer.encode(data_batch[2])
    ins = [
        data_batch[0], data_batch[1], data_batch[2], data_batch[3], data_batch[4],
        name_embeds.tolist(),
        descr_embeds.tolist()
    ]
    collection.insert(ins)

def process_csv_and_insert(file_path, collection, transformer):
    count = 0
    data_batch = [[], [], [], [], []]
    try:
        for col0, col1, col2, col3, col4 in csv_load(file_path):
            if count < env.COUNT:
                data_batch[0].append(int(col0))
                data_batch[1].append(col1)
                data_batch[2].append(col2)
                data_batch[3].append(int(col3))
                data_batch[4].append(col4)

                if len(data_batch[0]) % env.BATCH_SIZE == 0:
                    embed_insert(collection, data_batch, transformer)
                    data_batch = [[], [], [], [], []]
                count += 1
            else:
                break

        if len(data_batch[0]) > 0:
            embed_insert(collection, data_batch, transformer)
            count += len(data_batch[0])

        collection.flush()
        print('Inserted data successfully in:', collection.name)
        print('Number of inserted items:', count)
    except Exception as e:
        print('Error occurred during data insertion:', str(e))
        raise e

if __name__ == "__main__":
    connect_to_milvus()
    drop_collection_if_exists(env.PROJECT_COLLECTION_NAME)

    schema = create_collection_schema()
    collection = create_collection(env.PROJECT_COLLECTION_NAME, schema)
    process_csv_and_insert(env.PROJECT_FILE_PATH, collection, transformer)
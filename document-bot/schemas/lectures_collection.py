import csv
import warnings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import strip_markdown
import configs.env as env

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")


def connect_to_milvus():
    connections.connect(host=env.MILVUS_HOST, port=env.MILVUS_PORT)
    print('Connected to Milvus')

def drop_collection_if_exists(collection_name):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

def create_collection_schema():
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name='difficulty', dtype=DataType.INT32),
        FieldSchema(name='min_recommended_age', dtype=DataType.INT32),
        FieldSchema(name='max_recommended_age', dtype=DataType.INT32),
        FieldSchema(name='creator_id', dtype=DataType.INT64),
        FieldSchema(name='chunk_id', dtype=DataType.INT32),
        FieldSchema(name='name_emb', dtype=DataType.FLOAT_VECTOR, dim=env.DIMENSION),
        FieldSchema(name='content_emb', dtype=DataType.FLOAT_VECTOR, dim=env.DIMENSION),
    ]
    return CollectionSchema(fields)

def create_collection(collection_name, schema):
    collection = Collection(name=collection_name, schema=schema)

    collection.create_index(field_name='name_emb', index_params={'metric_type':'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
    collection.create_index(field_name='content_emb', index_params={'metric_type':'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
    collection.load()

    print('Collection created and indices created')
    return collection

def csv_load(file_path, splitter, chunk_size=256, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        data_chunk = []
        for row in reader:
            if len(row) < 6 or '' in (row[0], row[1], row[2], row[3], row[4], row[5]):
                continue
            name, content, difficulty, min_recommended_age, max_recommended_age, creator_id = row[0], row[1], row[2], row[3], row[4], row[5]
            split_result = splitter.split_text(text=strip_markdown.strip_markdown(content))
            chunk_id = 1 
            for chunk in split_result:
                data_chunk.append((name, chunk, difficulty, min_recommended_age, max_recommended_age, creator_id, chunk_id))
                chunk_id += 1
                if len(data_chunk) >= chunk_size:
                    yield data_chunk
                    data_chunk = []
        if data_chunk:
            yield data_chunk

def embed_insert(collection, data_batch, transformer):
    name_embeds = transformer.encode(data_batch[0])
    descr_embeds = transformer.encode(data_batch[1])
    ins = [
        data_batch[0], data_batch[1], data_batch[2], data_batch[3], data_batch[4], data_batch[5], data_batch[6],
        name_embeds.tolist(),
        descr_embeds.tolist()
    ]
    collection.insert(ins)

def process_csv_and_insert(file_path, collection, splitter, transformer):
    count = 0
    data_batch = [[], [], [], [], [], [], []]
    try:
        for row in csv_load(file_path, splitter):
            for col0, col1, col2, col3, col4, col5, col6 in row:
                if count < env.COUNT:
                    data_batch[0].append(col0)
                    data_batch[1].append(col1)
                    data_batch[2].append(int(col2))
                    data_batch[3].append(int(col3))
                    data_batch[4].append(int(col4))
                    data_batch[5].append(int(col5))
                    data_batch[6].append(int(col6))

                    if len(data_batch[0]) % env.BATCH_SIZE == 0:
                        embed_insert(collection, data_batch, transformer)
                        data_batch = [[], [], [], [], [], [], []]
                    count += 1
                else:
                    break

        if len(data_batch[0]) > 0:
            embed_insert(collection, data_batch, transformer)
            count += len(data_batch[0])

        collection.flush()
        print('Inserted data successfully in:', env.LECTURES_COLLECTION_NAME)
        print('Number of inserted items:', count)
    except Exception as e:
        print('Error occurred during data insertion:', str(e))
        raise e
    
def main():
    connect_to_milvus()
    drop_collection_if_exists(env.LECTURES_COLLECTION_NAME)

    splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10, model_name='sentence-transformers/all-mpnet-base-v2', tokens_per_chunk=128)
    transformer = SentenceTransformer('all-MiniLM-L6-v2')

    schema = create_collection_schema()
    collection = create_collection(env.LECTURES_COLLECTION_NAME, schema)
    process_csv_and_insert(env.LECTURES_FILE_PATH, collection, splitter, transformer)

if __name__ == "__main__":
    main()
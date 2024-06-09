PROJECT_COLLECTION_NAME = 'projects'
DOCUMENT_COLLECTION_NAME = 'documents'
LECTURES_COLLECTION_NAME = 'lectures'

DIMENSION = 384

# FIXME when running local
# MILVUS_HOST = '127.0.0.1'
MILVUS_HOST = 'milvus-standalone'
MILVUS_PORT = 19530
BATCH_SIZE = 128
TOP_K = 5
COUNT = 10000

PROJECT_FILE_PATH = "../data/projects.csv"
DOCUMENT_FILE_PATH = "../data/documents.csv"
LECTURES_FILE_PATH = "../data/lectures.csv"
import warnings

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")


splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=5, model_name='sentence-transformers/all-mpnet-base-v2', tokens_per_chunk=128)
transformer = SentenceTransformer('all-MiniLM-L6-v2')

def embed_insert(data):
    return transformer.encode(data).tolist()

def embed_search(data):
    return transformer.encode([data])

def chunk(text):
    data_chunks = []
    split_result = splitter.split_text(text=text)
    chunk_id = 1 
    for chunk_txt in split_result:
        data_chunks.append((chunk_id, chunk_txt))
        chunk_id += 1
    return data_chunks

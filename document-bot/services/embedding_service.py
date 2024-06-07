import warnings

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")


splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10, model_name='sentence-transformers/all-mpnet-base-v2', tokens_per_chunk=128)
transformer = SentenceTransformer('all-MiniLM-L6-v2')

def embed(data):
    return transformer.encode(data).tolist()
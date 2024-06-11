import io
import os
import re

import pymupdf
import requests
from configs.env import DOCUMENT_COLLECTION_NAME
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.embedding_service import chunk, embed_insert, embed_search
from services.milvus_service import client

router = APIRouter()

class VectorCreateRequest(BaseModel):
    project_id: int
    document_id: int
    name: str
    version: str
    text: str

class VectorResponse(BaseModel):
    id: int
    project_id: int
    document_id: int
    name: str
    version: str
    chunk_id: int
    text: str

class VectorSummarizeRequest(BaseModel):
    project_id: int
    document_id: int
    version: int

class VectorAskRequest(BaseModel):
    question: str
    project_id: int
    document_id: int
    version: int

class Vector(BaseModel):
    project_id: int
    document_id: int
    name: str
    version: int
    text: str
    chunk_id: int
    txt_emb: list

class NoVersionFound(Exception):
    pass

collection_name = DOCUMENT_COLLECTION_NAME

@router.get("/test-milvus-connection/")
async def test_milvus_connection():
    try:
        status = client.get_collection_stats(collection_name="documents")
        return {"message": "Connected to Milvus", "status": status}
    except Exception as e:
        return {"message": "Error occurred during Milvus connection:", "error": str(e)}

# FEAT: CRUD
@router.post("/api/v1/collections/documents")
async def create(
    project_id: int = Form(...),
    document_id: int = Form(...),
    name: str = Form(...),
    version: str = Form(...),
    text: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        if not text and not file:
            return JSONResponse(content={"message": "Either 'text' or 'file' must be provided"}, status_code=400)        
            
        if file and file.filename.endswith(".pdf"):
            text = pdf_to_text(file)

        for chunk_id, chunk_txt in chunk(text):
            vector = Vector(
                project_id=project_id,
                document_id=document_id,
                name=name,
                version=version,
                text=chunk_txt,
                chunk_id=chunk_id,
                txt_emb=embed_insert(chunk_txt)
            )
            client.insert(collection_name=collection_name, data=vector.dict())
        
        return JSONResponse(content={"message": "Vector(s) successfully created."}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# FEAT: SIMPLE QUERY 1 - Retrieval by ID
@router.get("/api/v1/collections/documents/{vector_id}", response_model=VectorResponse)
async def get(vector_id: int):
    try:
        vector_data = client.get(collection_name=collection_name, ids=vector_id)
        if vector_data:
            vector = vector_data[0]
            return VectorResponse(
                id=vector["id"],
                project_id=vector["project_id"],
                document_id=vector["document_id"],
                name=vector["name"],
                version=vector["version"],
                chunk_id=vector["chunk_id"],
                text=vector["text"]
            )
        else:
            return JSONResponse(content={"message": "Vector not found"}, status_code=404)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   
    
@router.delete("/api/v1/collections/documents/{vector_id}")
async def delete(vector_id: int):
    try:
        client.delete(collection_name=collection_name, pks=vector_id)
        return JSONResponse(content={"message": f"Vector with ID {vector_id} successfully deleted."}, status_code=200)
   
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# FEAT: SIMPLE QUERY 2 - Query with filters 
@router.post("/api/v1/collections/documents/summarize")
async def summarize(summarize_req: VectorSummarizeRequest):
    try:
        return summarize_document(summarize_req.project_id, 
                                  summarize_req.document_id,
                                  summarize_req.version)        
    except NoVersionFound as e:
        return JSONResponse(content={"error": str(e)}, status_code=404) 
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) 
    
# FEAT: COMPLEX QUERY 3 - Vector Search with filters 
@router.post("/api/v1/collections/documents/ask")
async def find(ask_req: VectorAskRequest):
    try:
        return answer_question(ask_req.question,
                               ask_req.project_id, 
                               ask_req.document_id,
                               ask_req.version)  
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)   

def pdf_to_text(file: UploadFile):
    try:
        pdf_data = io.BytesIO(file.file.read())
        pdf_document = pymupdf.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    
    except Exception as e:
        return JSONResponse(content={"error processing pdf": str(e)}, status_code=400)   

def find_documents_by_project(project_id):
    query_results = client.query(
        collection_name=collection_name,
        filter=f'project_id == {project_id}',
        output_fields=["project_id", "document_id","name", 'version'],
    )

    distinct_combos = set()
    documents = []
    for result in query_results:
        combo = (result['project_id'], result['document_id'], result['version'])
        if combo not in distinct_combos:
            distinct_combos.add(combo)
            documents.append(result)

    return documents

def find_chunks_by_version(project_id, document_id, version):
    return client.query(
        collection_name=collection_name,
        filter=f'project_id == {project_id} and document_id == {document_id} and version == {version}',
        output_fields=["chunk_id", "text"],
        sort_by="chunk_id"
    )   

def summarize_document(project_id, document_id, version, prompt = 'Shortly summarize the following document:'):
    chunks = find_chunks_by_version(project_id, document_id, version)
    
    if len(chunks) == 0:
        raise NoVersionFound(f"No vectors found for version {version}, document ID {document_id}, and project ID {project_id}")

    context = "\n\n".join(chunk['text'] for chunk in chunks)

    summary = run_prompt(prompt, context)
    return summary

def find_answer_by_version(question, project_id, document_id, version):
    text_vector = embed_search(question)

    return client.search(
        collection_name=collection_name,
        data=text_vector,
        filter=f'project_id == {project_id} and document_id == {document_id} and version == {version}',
        output_fields=["text"],
        limit=3
    )

def answer_question(prompt, project_id, document_id, version):
    chunks = find_answer_by_version(prompt, project_id, document_id, version)

    if len(chunks) == 0:
        raise NoVersionFound(f"No vectors found for version {version}, document ID {document_id}, and project ID {project_id}")

    context = "\n\n".join(chunk[0]['entity']['text'] for chunk in chunks)

    summary = run_prompt(prompt, context)
    return summary

def run_prompt(query, context):
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    api_token = os.environ['DOCUBOT_ACCESS_TOKEN']
    headers = {"Authorization": f"Bearer {api_token}"}

    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{query}\n\n{context}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    data = {
        "inputs": prompt,
        "parameters": {"max_length": 500, "min_length": 30, "do_sample": False}
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        generated_text = response.json()[0]['generated_text']
        response_text = generated_text.split('<|end_header_id|>\n\n')[1]
        return clean_response(response_text)
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def clean_response(response_text):
    clean_text = re.sub(r'\n\d\.', '', response_text)
    clean_text = re.sub(r'\n', '', clean_text)
    clean_text = re.sub(r'\*\*Background\*\*(.*)', '', clean_text)
    clean_text = re.sub(r'\*\*(.*?)\*\*', '', clean_text)
    return clean_text

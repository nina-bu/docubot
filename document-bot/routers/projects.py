import uvicorn
import configs.env as env
from fastapi import APIRouter
from services.milvus_service import client

router = APIRouter()

@router.get('/test-connection/')
async def test_connection():
    try:
        status = client.get_collection_stats(collection_name=env.PROJECT_COLLECTION)
        return {'message': 'Connected to Milvus', 'status': status}
    except Exception as e:
        return {'message': 'Error occurred during Milvus connection', 'error': str(e)}
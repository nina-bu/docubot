from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from datetime import datetime
import json

class EEventSource(str, Enum):
    ORCHESTRATOR = 'ORCHESTRATOR'
    DOCUMENT_BOT_SERVICE = 'DOCUMENT_BOT_SERVICE'

class ESagaStatus(str, Enum):
    SUCCESS = 'SUCCESS'
    ROLLBACK_PENDING = 'ROLLBACK_PENDING'
    FAIL = 'FAIL'

class Document(BaseModel):
    projectId: Optional[int]
    documentId: Optional[int]
    name: Optional[str]
    version: Optional[int]
    text: Optional[str]

class History(BaseModel):
    source: EEventSource
    status: ESagaStatus
    message: Optional[str]
    createdAt: datetime

class Event(BaseModel):
    id: Optional[str]
    transactionId: Optional[str]
    orderId: Optional[str]
    payload: Optional[Document]
    source: EEventSource
    createdAt: datetime
    status: ESagaStatus
    eventHistory: List[History] = []


def parse_event(json_str: str) -> Event:
    event_dict = json.loads(json_str)
    return Event(**event_dict)

def to_json(event: Event) -> str:
    return json.dumps(event, cls=CustomJSONEncoder)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Event):
            return obj.__dict__
        if isinstance(obj, Document):
            return obj.__dict__
        if isinstance(obj, History):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)
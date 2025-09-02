from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from ai_agency.engine import router as flow_router

router = APIRouter()

class ChatMessage(BaseModel):
    client_id: str
    session_id: str
    message: str

@router.post("/message_lead_capture")
async def handle_web_message(msg: ChatMessage):
    response = await flow_router.handle_message(
        client_id=msg.client_id,
        channel="web",
        purpose="lead_capture",
        session_id=msg.session_id,
        message=msg.message,
    )
    return {"reply": response}

@router.post("/message_support")
async def handle_web_message(msg: ChatMessage):
    response = await flow_router.handle_message(
        client_id=msg.client_id,
        channel="web",
        purpose="support",
        session_id=msg.session_id,
        message=msg.message,
    )
    return {"reply": response}
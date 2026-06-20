from fastapi import APIRouter

router = APIRouter(prefix="/chat")


@router.get("/status")
async def chat_status():
    return {"status": "ok"}

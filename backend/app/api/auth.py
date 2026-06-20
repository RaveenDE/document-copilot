from fastapi import APIRouter

router = APIRouter(prefix="/auth")


@router.get("/me")
async def get_current_user():
    return {"user": None}

from fastapi import APIRouter
from modules.storage.manager import Storage

router = APIRouter()
storage = Storage()

@router.get('/docs')
async def get_docs():
    return storage.get_docs()

@router.patch('/docs')
async def edit_docs(req):
    pass
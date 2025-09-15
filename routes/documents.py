from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.storage import Storage

router = APIRouter()
storage = Storage(dimension=1536)

class DocumentCreate(BaseModel):
    documents: list[str]

class DocumentUpdate(BaseModel):
    document: str


# Добавляет новые документы и возвращает их индексы
@router.post("/documents", response_model=list[int])
async def add_documents(doc_create: DocumentCreate):
    if not doc_create.documents:
        raise HTTPException(status_code=400, detail="Список документов пуст")
    indices = storage.push_docs(doc_create.documents)
    if not indices:
        raise HTTPException(status_code=500, detail="Ошибка при добавлении документов")
    return indices


# Возвращает все документы
@router.get("/documents", response_model=list[str])
async def get_documents():
    docs = storage.get_docs()
    return docs


# Ищет документы по запросу
@router.get("/documents/search", response_model=list[str])
async def search_documents(query: str, count: int = 5):
    if not query:
        raise HTTPException(status_code=400, detail="Запрос пуст")
    results = storage.search(query, count)
    if not results:
        raise HTTPException(status_code=500, detail="Ошибка при поиске документов")
    return results


# Возвращает документ по индексу
@router.get("/documents/{doc_id}", response_model=str)
async def get_document_by_id(doc_id: int):
    doc = storage.get_doc_by_id(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Документ с индексом {doc_id} не найден")
    return doc


# Обновляет документ по индексу
@router.put("/documents/{doc_id}", response_model=bool)
async def update_document(doc_id: int, doc_update: DocumentUpdate):
    success = storage.update_doc(doc_id, doc_update.document)
    if not success:
        raise HTTPException(status_code=400, detail=f"Ошибка при обновлении документа с индексом {doc_id}")
    return success


# Удаляет документ по индексу
@router.delete("/documents/{doc_id}", response_model=bool)
async def delete_document(doc_id: int):
    success = storage.delete_doc(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Документ с индексом {doc_id} не найден")
    return success
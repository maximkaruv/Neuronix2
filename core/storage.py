from loguru import logger
import faiss
import numpy as np
from openai import OpenAIError
from .embeddings.openai_embeddings import Embeddings
from .vectorstore.faiss_base import VectorBase

class Storage:
    def __init__(
            self,
            dimension: int = 1536,
            vectors_file: str = 'data/vectors.index',
            docs_file: str = 'data/documents.index'
        ):
        self.dimension = dimension
        self.database = VectorBase(dimension)
        self.vectors_file = vectors_file
        self.docs_file = docs_file
        self.embedding_model = Embeddings()
    
    # Загружает базы из путей
    def load(self) -> VectorBase:
        try:
            self.database = VectorBase.load(self.dimension, self.vectors_file, self.docs_file)
            return self.database
        except FileNotFoundError as e:
            logger.error(f"Ошибка загрузки: {e}. Создаем новую базу")
            self.database = VectorBase(self.dimension)
            return self.database
    
    # Сохраняем базу по путям
    def save(self):
        self.database.save(self.vectors_file, self.docs_file)
        logger.success("База сохранена")

    # Добавляет документы и возвращает их индексы
    def push_docs(self, docs: list[str]) -> list[int]:
        if not docs:
            logger.warning("Список документов пуст.")
            return []
        try:
            embeddings_list = self.embedding_model.vectorizate(docs, dimension=self.dimension)
            if len(embeddings_list) != len(docs):
                raise ValueError(f"Получено {len(embeddings_list)} эмбеддингов для {len(docs)} документов")
            start_idx = len(self.database.documents)
            self.database.add_embeddings_docs(embeddings_list, docs)
            return list(range(start_idx, start_idx + len(docs)))
        except OpenAIError as e:
            logger.error(f"Ошибка OpenAI-API: {e}")
            return []
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            return []

    # Ищет документы по запросу
    def search(self, query: str, count: int) -> list[str]:
        try:
            embeddings = self.embedding_model.vectorizate(query, dimension=self.dimension)
            embeddings = np.array(embeddings).astype('float32')
            if len(embeddings) != self.dimension:
                raise ValueError(f"Эмбеддинг запроса имеет размерность {len(embeddings)}, ожидается {self.dimension}")
            return self.database.search_neighbours(embeddings, count)
        except OpenAIError as e:
            logger.error(f"Ошибка OpenAI-API: {e}")
            return []
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    # Возвращает все документы
    def get_docs(self) -> list[str]:
        return self.database.documents

    # Возвращает документ по индексу
    def get_doc_by_id(self, doc_id: int) -> dict:
        if 0 <= doc_id < len(self.database.documents):
            return self.database.documents[doc_id]
        return None

    # Обновляет документ по индексу
    def update_doc(self, doc_id: int, new_doc: str) -> bool:
        if not (0 <= doc_id < len(self.database.documents)):
            logger.warning(f"Документ с индексом {doc_id} не существует.")
            return False
        try:
            embedding = self.embedding_model.vectorizate(new_doc, dimension=self.dimension)
            embedding = np.array(embedding).astype('float32')
            if len(embedding) != self.dimension:
                raise ValueError(f"Эмбеддинг нового документа имеет размерность {len(embedding)}, ожидается {self.dimension}")
            
            # Обновляем документ
            self.database.documents[doc_id] = new_doc
            # Обновляем эмбеддинг (FAISS не поддерживает прямое обновление, пересоздаём индекс)
            self._rebuild_index()
            return True
        except OpenAIError as e:
            logger.error(f"Ошибка OpenAI-API: {e}")
            return False
        except Exception as e:
            logger.error(f"Ошибка при обновлении документа: {e}")
            return False

    # Удаляет документ по индексу
    def delete_doc(self, doc_id: int) -> bool:
        if not (0 <= doc_id < len(self.database.documents)):
            logger.error(f"Документ с индексом {doc_id} не существует.")
            return False
        try:
            self.database.documents.pop(doc_id)
            # Пересоздаём индекс без удалённого документа
            self._rebuild_index()
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}")
            return False

    # Пересоздаёт FAISS индекс на основе текущих документов
    def _rebuild_index(self):
        try:
            embeddings_list = self.embedding_model.vectorizate(self.database.documents, dimension=self.dimension)
            self.database.index = faiss.IndexFlatL2(self.dimension)
            if embeddings_list:
                embeddings_np = np.array(embeddings_list).astype('float32')
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings_np = embeddings_np / norms
                self.database.index.add(embeddings_np)
        except OpenAIError as e:
            logger.error(f"Ошибка OpenAI-API при пересоздании индекса: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при пересоздании индекса: {e}")
            raise

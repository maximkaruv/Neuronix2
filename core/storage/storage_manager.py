import faiss
import numpy as np
from openai import OpenAIError
from core.models.embeddings import Embeddings
from core.storage.vectorbase import VectorBase
from typing import List, Optional

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
    
    def load(self) -> VectorBase:
        try:
            self.database = VectorBase.load(self.dimension, self.vectors_file, self.docs_file)
            return self.database
        except FileNotFoundError as e:
            print(f"Ошибка загрузки: {e}. Создаётся новая база.")
            self.database = VectorBase(self.dimension)
            return self.database
    
    def save(self):
        self.database.save(self.vectors_file, self.docs_file)

    def push_docs(self, docs: List[str]) -> List[int]:
        """Добавляет документы и возвращает их индексы."""
        if not docs:
            print("Предупреждение: Список документов пуст.")
            return []
        try:
            embeddings_list = self.embedding_model.vectorizate(docs, dimension=self.dimension)
            if len(embeddings_list) != len(docs):
                raise ValueError(f"Получено {len(embeddings_list)} эмбеддингов для {len(docs)} документов")
            start_idx = len(self.database.documents)
            self.database.add_embeddings_docs(embeddings_list, docs)
            return list(range(start_idx, start_idx + len(docs)))
        except OpenAIError as e:
            print(f"Ошибка OpenAI API: {e}")
            return []
        except Exception as e:
            print(f"Ошибка при добавлении документов: {e}")
            return []

    def search(self, query: str, count: int) -> List[str]:
        """Ищет документы по запросу."""
        try:
            embeddings = self.embedding_model.vectorizate(query, dimension=self.dimension)
            embeddings = np.array(embeddings).astype('float32')
            if len(embeddings) != self.dimension:
                raise ValueError(f"Эмбеддинг запроса имеет размерность {len(embeddings)}, ожидается {self.dimension}")
            return self.database.search_neighbours(embeddings, count)
        except OpenAIError as e:
            print(f"Ошибка OpenAI API: {e}")
            return []
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return []

    def get_docs(self) -> List[str]:
        """Возвращает все документы."""
        return self.database.documents

    def get_doc_by_id(self, doc_id: int) -> Optional[str]:
        """Возвращает документ по индексу."""
        if 0 <= doc_id < len(self.database.documents):
            return self.database.documents[doc_id]
        return None

    def update_doc(self, doc_id: int, new_doc: str) -> bool:
        """Обновляет документ по индексу."""
        if not (0 <= doc_id < len(self.database.documents)):
            print(f"Ошибка: Документ с индексом {doc_id} не существует.")
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
            print(f"Ошибка OpenAI API: {e}")
            return False
        except Exception as e:
            print(f"Ошибка при обновлении документа: {e}")
            return False

    def delete_doc(self, doc_id: int) -> bool:
        """Удаляет документ по индексу."""
        if not (0 <= doc_id < len(self.database.documents)):
            print(f"Ошибка: Документ с индексом {doc_id} не существует.")
            return False
        try:
            self.database.documents.pop(doc_id)
            # Пересоздаём индекс без удалённого документа
            self._rebuild_index()
            return True
        except Exception as e:
            print(f"Ошибка при удалении документа: {e}")
            return False

    def _rebuild_index(self):
        """Пересоздаёт FAISS индекс на основе текущих документов."""
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
            print(f"Ошибка OpenAI API при пересоздании индекса: {e}")
            raise
        except Exception as e:
            print(f"Ошибка при пересоздании индекса: {e}")
            raise

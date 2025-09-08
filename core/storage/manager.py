import numpy as np
from openai import OpenAIError
from core.models.embeddings import Embeddings
from core.storage.vectorbase import VectorBase

class Storage:
    def __init__(
            self,
            dimension: int = 1536,  # Для text-embedding-3-small
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

    def search(self, query: str, count: int) -> list[str]:
        try:
            embeddings = self.embedding_model.vectorizate(query)
            embeddings = np.array(embeddings).astype('float32')
            if len(embeddings) != self.dimension:
                raise ValueError(f"Эмбеддинг запроса имеет размерность {len(embeddings)}, ожидается {self.dimension}")
            results = self.database.search_neighbours(embeddings, count)
            return results
        except OpenAIError as e:
            print(f"Ошибка OpenAI API: {e}")
            return []
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return []

    def get_docs(self) -> list[str]:
        return self.database.documents

    def push_docs(self, docs: list[str]):
        if not docs:
            print("Предупреждение: Список документов пуст.")
            return
        try:
            embeddings_list = self.embedding_model.vectorizate(docs)  # Батчевая обработка
            if len(embeddings_list) != len(docs):
                raise ValueError(f"Получено {len(embeddings_list)} эмбеддингов для {len(docs)} документов")
            self.database.add_embeddings_docs(embeddings_list, docs)
        except OpenAIError as e:
            print(f"Ошибка OpenAI API при добавлении документов: {e}")
        except Exception as e:
            print(f"Ошибка при добавлении документов: {e}")
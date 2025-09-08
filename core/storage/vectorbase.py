import faiss
import numpy as np
import pickle
import os

class VectorBase:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def add_embeddings_docs(self, embeddings: list, documents: list):
        if len(embeddings) != len(documents):
            raise ValueError(f"Количество эмбеддингов ({len(embeddings)}) и документов ({len(documents)}) должно совпадать.")
        
        embeddings_np = np.array(embeddings).astype('float32')
        if embeddings_np.shape[1] != self.index.d:
            raise ValueError(f"Размерность эмбеддингов ({embeddings_np.shape[1]}) не соответствует размерности индекса ({self.index.d})")
        
        # Нормализация с проверкой на нулевые векторы
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Избегаем деления на 0
        embeddings_np = embeddings_np / norms
        self.index.add(embeddings_np)
        self.documents.extend(documents)
    
    def search_neighbours(self, query_embedding: list, k: int = 5) -> list:
        query_np = np.array([query_embedding]).astype('float32')
        if query_np.shape[1] != self.index.d:
            raise ValueError(f"Размерность запроса ({query_np.shape[1]}) не соответствует размерности индекса ({self.index.d})")
        
        # Нормализация с проверкой
        norm = np.linalg.norm(query_np, axis=1, keepdims=True)
        norm[norm == 0] = 1
        query_np = query_np / norm
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append(self.documents[idx])
        
        return results
    
    def save(self, index_path: str, docs_path: str):
        if not os.path.exists(os.path.dirname(index_path)) and os.path.dirname(index_path):
            os.makedirs(os.path.dirname(index_path))
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load(cls, dimension: int, index_path: str, docs_path: str):
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"Файлы {index_path} или {docs_path} не найдены.")
        index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        
        instance = cls(dimension)
        instance.index = index
        instance.documents = documents
        
        return instance
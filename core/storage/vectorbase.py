import faiss
import numpy as np

class VectorBase:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    

    def add_embeddings(self, embeddings: list, documents: list):
        if len(embeddings) != len(documents):
            raise ValueError("Количество эмбеддингов и документов должно совпадать.")
        
        embeddings_np = np.array(embeddings).astype('float32')
        
        self.index.add(embeddings_np)
        self.documents.extend(documents)
    

    def search_neighbours(self, query_embedding: list, k: int = 5) -> list:
        query_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                # results.append((self.documents[idx], distances[0][i]))
                results.append(self.documents[idx])
        
        return results
    

    def save(self, index_path: str, docs_path: str):
        import pickle
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    
    @classmethod
    def load(cls, dimension: int, index_path: str, docs_path: str):
        import pickle
        index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        
        instance = cls(dimension)
        instance.index = index
        instance.documents = documents
        
        return instance

from modules.core.embeddings import vectorizate
from modules.core.vectorbase import VectorBase

class Storage:
    def __init__(
            self,
            dimension: int = 1024, 
            vectors_file: str = 'data/vectors.index', 
            docs_file: str = 'data/documents.index'
        ):
        self.dimension = dimension
        self.database = VectorBase(dimension)
        self.vectors_file = vectors_file
        self.docs_file = docs_file
    
    def load(self) -> VectorBase:
        self.database = VectorBase.load(self.dimension, self.vectors_file, self.docs_file)
    
    def save(self):
        self.database.save(self.vectors_file, self.docs_file)

    def search(self, query: str, count: int) -> list[str]:
        embeddings: list[int] = vectorizate(query)
        results = self.database.search_neighbours(embeddings, count)
        return results
    
    def get_docs(self):
        return []

    def push_docs(self, docs: list[str]):
        for doc in docs:
            embeddings: list[int] = vectorizate(doc)
            self.database.add_embeddings(embeddings, doc)
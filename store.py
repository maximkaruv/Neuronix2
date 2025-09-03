import faiss
import numpy as np

class FaissVectorStore:
    """
    Простой модуль для работы с FAISS в контексте RAG-системы.
    Позволяет добавлять эмбеддинги документов, сохранять/загружать индекс и выполнять поиск по сходству.
    
    Аргументы:
    - dimension: размерность эмбеддингов (например, 768 для BERT-like моделей).
    
    Методы:
    - add_embeddings: добавляет эмбеддинги и соответствующие документы (тексты или метаданные).
    - search: выполняет поиск ближайших соседей по запросному эмбеддингу.
    - save: сохраняет индекс и метаданные на диск.
    - load: загружает индекс и метаданные с диска.
    """
    
    def __init__(self, dimension: int):
        # Используем простой плоский индекс с L2-метрикой (евклидово расстояние)
        self.index = faiss.IndexFlatL2(dimension)
        # Список для хранения оригинальных документов (текстов или метаданных)
        self.documents = []
    
    def add_embeddings(self, embeddings: list, documents: list):
        """
        Добавляет эмбеддинги и соответствующие документы в индекс.
        
        Аргументы:
        - embeddings: список numpy-массивов или list of lists с эмбеддингами.
        - documents: список строк или словарей с метаданными (должен быть той же длины, что и embeddings).
        """
        if len(embeddings) != len(documents):
            raise ValueError("Количество эмбеддингов и документов должно совпадать.")
        
        # Преобразуем в numpy-массив float32 для FAISS
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Добавляем в индекс
        self.index.add(embeddings_np)
        
        # Сохраняем документы
        self.documents.extend(documents)
    
    def search(self, query_embedding: list, k: int = 5) -> list:
        """
        Выполняет поиск k ближайших документов по запросному эмбеддингу.
        
        Аргументы:
        - query_embedding: эмбеддинг запроса (list или numpy-array).
        - k: количество результатов для возврата.
        
        Возвращает: список кортежей (document, distance), отсортированных по расстоянию.
        """
        # Преобразуем запрос в numpy-массив
        query_np = np.array([query_embedding]).astype('float32')
        
        # Поиск в индексе
        distances, indices = self.index.search(query_np, k)
        
        # Собираем результаты, игнорируя -1 (если результатов меньше k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append((self.documents[idx], distances[0][i]))
        
        return results
    
    def save(self, index_path: str = 'faiss_index.index', docs_path: str = 'documents.pkl'):
        """
        Сохраняет индекс и документы на диск.
        """
        import pickle
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load(cls, dimension: int, index_path: str = 'faiss_index.index', docs_path: str = 'documents.pkl'):
        """
        Загружает индекс и документы с диска.
        """
        import pickle
        index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        
        instance = cls(dimension)
        instance.index = index
        instance.documents = documents
        return instance

# Пример использования
if __name__ == "__main__":
    # Предположим, у нас есть эмбеддинги размерностью 4 (для теста)
    dimension = 4
    store = FaissVectorStore(dimension)
    
    # Пример эмбеддингов и документов
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    documents = [
        "Документ 1: Это текст про AI.",
        "Документ 2: RAG системы используют retrieval.",
        "Документ 3: FAISS для векторного поиска."
    ]
    
    store.add_embeddings(embeddings, documents)
    
    # Пример поиска
    query = [0.4, 0.5, 0.6, 0.7]  # Близко ко второму эмбеддингу
    results = store.search(query, k=2)
    print("Результаты поиска:")
    for doc, dist in results:
        print(f"Документ: {doc}, Расстояние: {dist}")
    
    # Сохранение и загрузка
    store.save()
    loaded_store = FaissVectorStore.load(dimension)
    print("\nПосле загрузки:")
    results_loaded = loaded_store.search(query, k=2)
    for doc, dist in results_loaded:
        print(f"Документ: {doc}, Расстояние: {dist}")
from openai import OpenAI, OpenAIError
from config import openai_model

class Embeddings:
    def __init__(self):
        self.client = OpenAI(
            api_key=openai_model['api_key'],
            base_url=openai_model['base_url']
        )
    
    def vectorizate(self, text, verbose=True):
        try:
            res = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            if verbose:
                print("Получена векторизация: ", str(res.data[0].embedding)[:1000])
            # Для батчевого ввода возвращаем список эмбеддингов
            if isinstance(text, list):
                return [item.embedding for item in res.data]
            return res.data[0].embedding
        
        except OpenAIError as e:
            print(f"Ошибка OpenAI API: {e}")
            raise
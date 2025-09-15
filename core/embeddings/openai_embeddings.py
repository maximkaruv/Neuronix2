from loguru import logger
from openai import OpenAI, OpenAIError
from config import openai

class Embeddings:
    def __init__(self):
        self.client = OpenAI(
            api_key=openai['api_key'],
            base_url=openai['base_url']
        )
    
    def vectorizate(self, text, dimension=None, verbose=True):
        try:
            res = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            # Для батчевого ввода возвращаем список эмбеддингов
            if isinstance(text, list):
                return [item.embedding for item in res.data]
            return res.data[0].embedding
        
        except OpenAIError as e:
            logger.error(f"Ошибка векторизации OpenAI-API: {e}")
            raise
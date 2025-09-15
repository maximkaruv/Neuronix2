from loguru import logger
from openai import OpenAI, OpenAIError
from config import openai

class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=openai['api_key'],
            base_url=openai['base_url']
        )
    
    def prompt(self, text):
        try:
            res = self.client.embeddings.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": text}]
            )
            return res.choices[0].message.content
        except OpenAIError as e:
            logger.error(f"Ошибка генерации OpenAI-API: {e}")
            raise
from openai import OpenAI

class Embeddings:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def vectorizate(self, text):
        res = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return res.data[0].embedding
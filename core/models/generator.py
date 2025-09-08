from openai import OpenAI

class Generator:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def prompt(self, text):
        res = self.client.embeddings.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": text}]
        )
        return res.choices[0].message.content

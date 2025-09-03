from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('DeepPavlov/rubert-base-cased')

res = model.encode(["Привет, дружище, как дела у деда Мороза? Там ведь холодно, мороз"])
print(res, res[0])
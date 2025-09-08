from core.storage.manager import Storage

storage = Storage(dimension=1536)

print("Добавляем документы")
storage.push_docs(["Текст 1 про AI.", "Текст 2 про RAG."])
print("Сохраняем базу данных")
storage.save()
print("Ищем")
results = storage.search("Что такое RAG?", count=1)
print(results)
print("Документы в базе:", storage.get_docs())

# python -m modules.storage.py
from fastapi import APIRouter
from modules.generator import prompt
from modules.storage import Storage

router = APIRouter()
storage = Storage()

BLOCK = """Содержание: \"{content}\"
Источник: \"{source}\""""

MESSAGE = """Ты - поисковый помощник моей компании. Я хочу, чтобы ты нашел в следующих документах информацию, которую нужно использовать, чтобы дать ответ на вопрос, если информации недостаточно - не выдумывай и напиши \"Недостаточно информации\". Если информации достаточно то в ответе кратко изложи ответ на вопрос и для каждого пункта или предложения, где это необходимо, вставляй в скобочках источник данного факта - пример: (Отчёт о хозтоварах, стр. 5)
Вопрос: \"{query}\"
Блоки информации:
---
{blocks}
---"""

@router.post('/ask')
async def ask(req):
    query = req.query
    matches = storage.search(query, count=5)
    blocks = [BLOCK.format(content=block.content, source=block.source) for block in matches]
    answer = prompt(MESSAGE.format(matches='\n---\n'.join(blocks)))
    return answer
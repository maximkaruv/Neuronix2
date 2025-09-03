from fastapi import FastAPI
from routes import ask, store

app = FastAPI()

app.include_router(ask, prefix='/api')
app.include_router(store, prefix='/api')

@app.get('/')
async def ping():
    return {"status": "work"}
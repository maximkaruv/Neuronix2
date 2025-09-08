import sys, os
sys.pycache_prefix = os.path.join(os.getcwd(), "pycache")

from fastapi import FastAPI
from modules import ask, store

app = FastAPI()

app.include_router(ask, prefix='/api')
app.include_router(store, prefix='/api')

@app.get('/')
async def ping():
    return {"status": "work"}
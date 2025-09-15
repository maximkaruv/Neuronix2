import sys, os
sys.pycache_prefix = os.path.join(os.getcwd(), "__pycache__")

from fastapi import FastAPI
import uvicorn
import routes.ask as ask
import routes.storage as storage

app = FastAPI()

app.include_router(ask.router, prefix='/api')
app.include_router(storage.router, prefix='/api')

@app.get('/')
async def ping():
    return {"status": "working"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
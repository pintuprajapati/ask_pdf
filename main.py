from fastapi import FastAPI
from rag import create_persist_dir

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}   
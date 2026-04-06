from fastapi import FastAPI, UploadFile, File
from typing import List
import tempfile
import shutil
import os

from app.model import embedding_model

app = FastAPI()

@app.post("/embed")
async def embed_images(files: List[UploadFile] = File(...)):
    temp_paths = []

    try:
        # Save uploaded files temporarily
        for file in files:
            suffix = os.path.splitext(file.filename)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            with temp_file as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_paths.append(temp_file.name)

        # Generate embeddings
        embeddings = embedding_model.embed(temp_paths)

        # Convert numpy arrays to lists
        return {
            "embeddings": [emb.tolist() for emb in embeddings]
        }

    finally:
        # Cleanup temp files
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
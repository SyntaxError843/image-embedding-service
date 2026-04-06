from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import tempfile
import requests
import os

from app.model import embedding_model

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB

app = FastAPI()

class EmbedRequest(BaseModel):
    image_urls: List[HttpUrl]


@app.post("/embed")
def embed_images(request: EmbedRequest):
    if not request.image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")

    temp_paths = []

    try:
        # Download images
        for url in request.image_urls:
            response = requests.get(str(url), timeout=10)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: {url}"
                )
                
            if int(response.headers.get("content-length", 0)) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=400, detail="Image too large")
            
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="URL is not an image")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(response.content)
            temp_file.close()

            temp_paths.append(temp_file.name)

        # Generate embeddings
        embeddings = embedding_model.embed(temp_paths)

        return {
            "embeddings": [emb.tolist() for emb in embeddings]
        }

    finally:
        # Cleanup
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
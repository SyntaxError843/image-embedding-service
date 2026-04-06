from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import tempfile
import requests
import os

from app.model import embedding_model

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

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
            response = requests.get(str(url), headers=HEADERS, timeout=10)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: {url} {response.text}"
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
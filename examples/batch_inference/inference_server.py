import asyncio
import base64
from io import BytesIO
from typing import Dict, List

import clip
from fastapi import BackgroundTasks
from fastapi import FastAPI
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn


class ClipInferenceServer:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.lock = asyncio.Lock()

    async def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for a PIL Image"""
        # Preprocess the image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Use lock to prevent concurrent GPU operations
        async with self.lock:
            # Calculate the image embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)

            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy()


class ImageBatch(BaseModel):
    batch_id: str
    images: List[str]  # Base64 encoded images


app = FastAPI()
inference_server = ClipInferenceServer()


@app.post("/embed")
async def embed_images(batch: ImageBatch):
    # Convert base64 strings directly to PIL Images
    images = []
    valid_indices = []  # Keep track of which images were successfully processed

    for idx, img_data in enumerate(batch.images):
        try:
            image_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(image_bytes))
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error decoding image at index {idx}: {str(e)}")
            continue

    if not images:
        return {
            "batch_id": batch.batch_id,
            "embeddings": [],
            "valid_indices": []
        }

    # Process images concurrently
    try:
        embeddings = await asyncio.gather(
            *[inference_server.get_embedding(img) for img in images],
            return_exceptions=True)

        # Filter out failed embeddings
        valid_embeddings = []
        final_indices = []
        for idx, emb in enumerate(embeddings):
            if not isinstance(emb, Exception):
                valid_embeddings.append(emb.tolist())
                final_indices.append(valid_indices[idx])

        return {
            "batch_id": batch.batch_id,
            "embeddings": valid_embeddings,
            "valid_indices": final_indices
        }
    except Exception as e:
        print(f"Error processing batch {batch.batch_id}: {str(e)}")
        return {
            "batch_id": batch.batch_id,
            "embeddings": [],
            "valid_indices": []
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

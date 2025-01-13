import asyncio
import base64
from io import BytesIO
from typing import Dict, List, Union

import open_clip
from fastapi import FastAPI
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]  # List of base64 encoded images
    model: str = "ViT-B/32"  # Default model, similar to OpenAI's model parameter


class ClipInferenceServer:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}  # Dictionary to store loaded models
        self.lock = asyncio.Lock()

    async def get_model(self, model_name: str):
        """Load model if not already loaded"""
        if model_name not in self.models:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained="laion2b_s39b_b160k",  
                device=self.device
            )
            self.models[model_name] = (model, preprocess)
        return self.models[model_name]

    async def get_embedding(self, image: Image.Image, model_name: str) -> np.ndarray:
        """Get CLIP embedding for a PIL Image using specified model"""
        # Get or load the requested model
        model, preprocess = await self.get_model(model_name)
        
        # Preprocess the image
        image_input = preprocess(image).unsqueeze(0).to(self.device)

        # Use lock to prevent concurrent GPU operations
        async with self.lock:
            # Calculate the image embedding
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy()


app = FastAPI()
inference_server = ClipInferenceServer()


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    # Handle both single string and list inputs
    if isinstance(request.input, str):
        images_data = [request.input]
    else:
        images_data = request.input

    # Convert base64 strings directly to PIL Images
    images = []
    valid_indices = []

    for idx, img_data in enumerate(images_data):
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
            "object": "list",
            "data": [],
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }

    # Process images concurrently with specified model
    try:
        embeddings = await asyncio.gather(
            *[inference_server.get_embedding(img, request.model) for img in images],
            return_exceptions=True)

        # Format response similar to OpenAI
        data = []
        for idx, emb in enumerate(embeddings):
            if not isinstance(emb, Exception):
                data.append({
                    "object": "embedding",
                    "embedding": emb.flatten().tolist(),
                    "index": valid_indices[idx]
                })

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": len(
                    data
                ),  # Using number of successful embeddings as token count
                "total_tokens": len(data)
            }
        }
    except Exception as e:
        print(f"Error processing embeddings: {str(e)}")
        return {
            "object": "list",
            "data": [],
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)

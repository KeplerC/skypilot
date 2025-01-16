import asyncio
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Union

from fastapi import FastAPI
import numpy as np
import open_clip
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]  # List of base64 encoded images
    model: str = "ViT-B/32"  # Default model


class BatchEmbeddingRequest(BaseModel):
    inputs: List[Dict[
        str, str]]  # List of requests, each with input and model fields


class ClipInferenceServer:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}  # Dictionary to store loaded models
        self.lock = asyncio.Lock()

    async def get_model(self, model_name: str):
        """Load model if not already loaded"""
        if model_name not in self.models:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="laion2b_s39b_b160k", device=self.device)
            self.models[model_name] = (model, preprocess)
        return self.models[model_name]

    async def get_embedding(self, images: List[Image.Image],
                            model_name: str) -> np.ndarray:
        """Get CLIP embeddings for a batch of PIL Images using specified model"""
        # Get or load the requested model
        model, preprocess = await self.get_model(model_name)

        # Preprocess all images in the batch
        image_inputs = torch.stack([preprocess(img) for img in images
                                   ]).to(self.device)

        # Use lock to prevent concurrent GPU operations
        async with self.lock:
            # Calculate the image embeddings in batch
            with torch.no_grad():
                image_features = model.encode_image(image_inputs)

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

    # Process all images in a single batch
    try:
        embeddings = await inference_server.get_embedding(images, request.model)

        # Format response similar to OpenAI
        data = [{
            "object": "embedding",
            "embedding": emb.flatten().tolist(),
            "index": idx
        } for idx, emb in zip(valid_indices, embeddings)]

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": len(data),
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


@app.post("/v1/embeddings/batch")
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    # Group requests by model to process efficiently
    model_groups: Dict[str, List[Tuple[int, str]]] = {}
    for idx, req in enumerate(request.inputs):
        model_name = req.get("model", "ViT-B/32")
        image_data = req["input"]
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append((idx, image_data))

    all_results = []

    # Process each model group
    for model_name, image_group in model_groups.items():
        indices, images_data = zip(*image_group)

        # Convert base64 strings to PIL Images
        images = []
        valid_indices = []

        for idx, img_data in zip(indices, images_data):
            try:
                image_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(image_bytes))
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error decoding image at index {idx}: {str(e)}")
                continue

        if images:
            try:
                embeddings = await inference_server.get_embedding(
                    images, model_name)

                # Create results for this batch
                results = [{
                    "object": "embedding",
                    "embedding": emb.flatten().tolist(),
                    "index": idx
                } for idx, emb in zip(valid_indices, embeddings)]
                all_results.extend(results)
            except Exception as e:
                print(
                    f"Error processing embeddings for model {model_name}: {str(e)}"
                )

    # Sort results by original index
    all_results.sort(key=lambda x: x["index"])

    return {
        "object": "list",
        "data": all_results,
        "model": "batch",  # Indicate this was a batch request
        "usage": {
            "prompt_tokens": len(all_results),
            "total_tokens": len(all_results)
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)

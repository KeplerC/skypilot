import logging
from typing import Dict, Any, Optional, List

import numpy as np
import torch

from .inference import BatchInferenceProcessor
from .datasets import HuggingFaceDatasetMixin

class ClipBatchProcessor(HuggingFaceDatasetMixin, BatchInferenceProcessor[Dict[str, Any], np.ndarray]):
    """Example implementation for processing images with CLIP."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        dataset_name: str = "laion/relaion2B-en-research-safe",
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            **kwargs
        )
        self.preprocess = None
        
    async def setup_model(self):
        """Set up the CLIP model."""
        import open_clip
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained="laion2b_s39b_b160k",
            device=self.device
        )
        self.model = model
        self.preprocess = preprocess
        
    async def preprocess_input(
        self,
        item: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Download and preprocess a single image."""
        import aiohttp
        from PIL import Image
        from io import BytesIO
        
        url = item["url"]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, ssl=False) as response:
                    if response.status == 200:
                        data = await response.read()
                        img = Image.open(BytesIO(data))
                        return self.preprocess(img)
        except Exception as e:
            logging.debug(f"Error preprocessing image from {url}: {str(e)}")
        return None
        
    async def run_model_inference(
        self,
        model_inputs: List[torch.Tensor]
    ) -> List[np.ndarray]:
        """Run CLIP model on a batch of preprocessed images."""
        # Stack inputs into a batch
        batch_tensor = torch.stack(model_inputs).to(self.device)
        
        # Run inference
        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)
            
        # Convert to numpy arrays
        return features.cpu().numpy() 
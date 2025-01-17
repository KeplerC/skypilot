import asyncio
import logging

import logging
from typing import Dict, Any, Optional, List, AsyncIterator, Tuple

import numpy as np
import torch

from processors import BatchInferenceProcessor
from processors  import HuggingFaceDatasetMixin

class ClipBatchProcessor(HuggingFaceDatasetMixin, BatchInferenceProcessor[Dict[str, Any], np.ndarray]):
    """Example implementation for processing images with CLIP."""
    
    def __init__(
        self,
        model_name: str = "ViT-bigG-14",
        dataset_name: str = "laion/relaion2B-en-research-safe",
        max_preprocessing_tasks: int = 10,  # Control parallel preprocessing
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            **kwargs
        )
        self.preprocess = None
        self.preprocessing_semaphore = asyncio.Semaphore(max_preprocessing_tasks)
        
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
    
    async def do_data_loading(self) -> AsyncIterator[Tuple[int, torch.Tensor]]:
        """Load and preprocess data in parallel."""
        if self.model is None:
            await self.setup_model()
            
        # Create a buffer to store preprocessing tasks
        preprocessing_tasks = []
        buffer_size = self.batch_size * 2  # Maintain 2 batches worth of preprocessing tasks
        
        async for idx, item in self.get_dataset_iterator():
            # Clean up completed tasks when buffer is full
            if len(preprocessing_tasks) >= buffer_size:
                # Wait for at least one task to complete
                done, pending = await asyncio.wait(
                    preprocessing_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                preprocessing_tasks = list(pending)
                
                # Yield completed results
                for task in done:
                    result = await task
                    if result[1] is not None:  # Only yield successfully preprocessed items
                        yield result

            # Start new preprocessing task
            async def preprocess_with_index(idx, item):
                async with self.preprocessing_semaphore:
                    tensor = await self._preprocess_input(item)
                    return (idx, tensor)
                    
            task = asyncio.create_task(preprocess_with_index(idx, item))
            preprocessing_tasks.append(task)
            
        # Wait for and yield remaining results
        if preprocessing_tasks:
            done = await asyncio.gather(*preprocessing_tasks)
            for result in done:
                if result[1] is not None:  # Only yield successfully preprocessed items
                    yield result

    async def _preprocess_input(
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
        
async def main():
    """Example usage of the batch processing framework."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize processor
    processor = ClipBatchProcessor(
        output_path="embeddings.parquet",
        start_idx=0,
        end_idx=1000,
        batch_size=50,
        checkpoint_size=100,
        model_name="ViT-bigG-14",
        dataset_name="laion/relaion2B-en-research-safe"
    )
    
    # Run processing
    await processor.run()

if __name__ == "__main__":
    asyncio.run(main()) 
import asyncio
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import torch
from PIL import Image

from base_vector_processor import BaseVectorProcessor

class ImageVectorProcessor(BaseVectorProcessor):
    """Process images with CLIP to compute vector embeddings."""

    def __init__(self,
                 output_path: str,
                 model_name: str = 'ViT-bigG-14',
                 dataset_name: str = 'laion/relaion2B-en-research-safe',
                 pretrained: str = 'laion2b_s39b_b160k',
                 device: Optional[str] = None,
                 split: str = 'train',
                 streaming: bool = True,
                 batch_size: int = 32,
                 checkpoint_size: int = 100,
                 start_idx: int = 0,
                 end_idx: Optional[int] = None,
                 max_preprocessing_tasks: int = 10):
        """Initialize the image vector processor.
        
        Args:
            output_path: Path to save the computed vectors
            model_name: Name of the CLIP model to use
            dataset_name: Name of the dataset to process
            pretrained: Pretrained weights to use
            device: Device to use for inference
            split: Dataset split to use
            streaming: Whether to stream the dataset
            batch_size: Size of batches for processing
            checkpoint_size: Number of items to process before saving
            start_idx: Starting index in the dataset
            end_idx: Ending index in the dataset
            max_preprocessing_tasks: Maximum number of concurrent preprocessing tasks
        """
        super().__init__(
            output_path=output_path,
            dataset_name=dataset_name,
            split=split,
            streaming=streaming,
            batch_size=batch_size,
            checkpoint_size=checkpoint_size,
            start_idx=start_idx,
            end_idx=end_idx,
            max_preprocessing_tasks=max_preprocessing_tasks
        )
        # CLIP-specific attributes
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess = None

    async def setup_model(self):
        """Set up the CLIP model."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device)
        self.model = model
        self.preprocess = preprocess
        logging.info(f"Loaded CLIP model {self.model_name} on {self.device}")

    async def get_dataset_iterator(self) -> AsyncIterator[Tuple[int, Any]]:
        """Load data from a HuggingFace dataset."""
        from datasets import load_dataset

        dataset = load_dataset(self.dataset_name,
                             streaming=self.streaming,
                             trust_remote_code=True)[self.split]

        if self.start_idx > 0:
            dataset = dataset.skip(self.start_idx)

        for idx, item in enumerate(dataset, start=self.start_idx):
            if self.end_idx and idx >= self.end_idx:
                break
            yield idx, item

    async def _preprocess_input(self, item: Any) -> Optional[Tuple[str, torch.Tensor]]:
        """Download and preprocess a single image from URL."""
        url = item["url"]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, ssl=False) as response:
                    if response.status == 200:
                        data = await response.read()
                        img = Image.open(BytesIO(data))
                        tensor = self.preprocess(img)
                        return (url, tensor)
        except Exception as e:
            self.failed_count += 1
            self.processed_count += 1  # Count failed items as processed
            logging.debug(f"Error preprocessing image from {url}: {str(e)}")
        return None

    async def do_batch_processing(
        self, batch: List[Tuple[int, Tuple[str, torch.Tensor]]]
    ) -> List[Tuple[int, bytes]]:
        """Process a batch of images through CLIP."""
        if self.model is None:
            await self.setup_model()

        # Unpack the batch
        indices, batch_data = zip(*batch)
        urls, tensors = zip(*batch_data)

        # Stack inputs into a batch
        batch_tensor = torch.stack(tensors).to(self.device)

        # Run inference
        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        # Convert to numpy and serialize
        result = []
        for idx, url, feature in zip(indices, urls, features):
            feature_np = feature.cpu().numpy().astype(np.float32)
            feature_bytes = feature_np.tobytes()
            
            result.append((idx, feature_bytes, url))
            self.processed_count += 1

        return result

    def save_results_to_parquet(self, results: List[Tuple[int, bytes, str]]):
        """Save results to a parquet file with partition."""
        if not results:
            return

        # Create a dataframe from the results
        df = pd.DataFrame({
            'idx': [r[0] for r in results],
            'embedding': [r[1] for r in results],
            'url': [r[2] for r in results]
        })

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_path.parent, exist_ok=True)

        # Generate the partition output path
        base_name = self.output_path.stem
        output_path = self.output_path.parent / f"{base_name}_part{self.partition_counter}.parquet"
        
        # Save to parquet
        df.to_parquet(output_path)
        logging.info(f"Saved {len(df)} embeddings to {output_path}")
        
        # Increment partition counter
        self.partition_counter += 1

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute image vectors using CLIP')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the output parquet file')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index in the dataset')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='Ending index in the dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--checkpoint-size', type=int, default=100,
                        help='Number of items to process before saving')
    parser.add_argument('--model-name', type=str, default='ViT-bigG-14',
                        help='CLIP model name')
    parser.add_argument('--dataset-name', type=str, 
                        default='laion/relaion2B-en-research-safe',
                        help='HuggingFace dataset name')
    parser.add_argument('--pretrained', type=str, default='laion2b_s39b_b160k',
                        help='Pretrained weights to use')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize processor
    processor = ImageVectorProcessor(
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size,
        checkpoint_size=args.checkpoint_size,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        pretrained=args.pretrained
    )

    # Run processing
    await processor.run()

if __name__ == '__main__':
    asyncio.run(main()) 
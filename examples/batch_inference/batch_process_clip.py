import abc
import asyncio
import logging
import os
import pickle
from typing import (Any, AsyncIterator, Dict, Generic, List, Optional, Tuple,
                    TypeVar)

import numpy as np
import torch
from tqdm import tqdm

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
ModelInputType = TypeVar('ModelInputType')
ModelOutputType = TypeVar('ModelOutputType')


class BatchProcessor(Generic[InputType, OutputType], abc.ABC):
    """Process ImageNet images with CLIP."""

    def __init__(self,
                 output_path: str,
                 model_name: str = "ViT-bigG-14",
                 dataset_name: str = "ILSVRC/imagenet-1k",
                 pretrained: str = "laion2b_s39b_b160k",
                 device: Optional[str] = None,
                 split: str = "train",
                 streaming: bool = True,
                 batch_size: int = 32,
                 checkpoint_size: int = 100,
                 start_idx: int = 0,
                 end_idx: Optional[int] = None):
        self.output_path = output_path
        self.batch_size = batch_size
        self.checkpoint_size = checkpoint_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self._current_batch = []

        # CLIP-specific attributes
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.model = None
        self.preprocess = None

    async def setup_model(self):
        """Set up the CLIP model."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device)
        self.model = model
        self.preprocess = preprocess

    async def get_dataset_iterator(self) -> AsyncIterator[Tuple[int, Any]]:
        """Load data from a HuggingFace dataset."""
        from datasets import load_dataset

        dataset = load_dataset(self.dataset_name,
                               streaming=self.streaming)[self.split]

        if self.start_idx > 0:
            dataset = dataset.skip(self.start_idx)

        for idx, item in enumerate(dataset, start=self.start_idx):
            if self.end_idx and idx >= self.end_idx:
                break
            yield idx, item

    async def do_data_loading(self) -> AsyncIterator[Tuple[int, torch.Tensor]]:
        """Load and preprocess ImageNet images."""
        if self.model is None:
            await self.setup_model()

        async for idx, item in self.get_dataset_iterator():
            try:
                # ImageNet provides PIL Images directly
                tensor = self.preprocess(item['image'])
                if tensor is not None:
                    yield idx, tensor
            except Exception as e:
                logging.debug(
                    f"Error preprocessing image at index {idx}: {str(e)}")

    async def do_batch_processing(
            self, batch: List[Tuple[int,
                                    torch.Tensor]]) -> List[Tuple[int, bytes]]:
        """Process a batch of images through CLIP."""
        if self.model is None:
            await self.setup_model()

        indices, model_inputs = zip(*batch)

        # Stack inputs into a batch
        batch_tensor = torch.stack(model_inputs).to(self.device)

        # Run inference
        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        # Convert to numpy arrays and then to bytes
        embeddings = features.cpu().numpy()
        return list(zip(indices, [pickle.dumps(arr) for arr in embeddings]))

    async def run(self):
        """Run the batch processing pipeline."""
        import pandas as pd

        results = []
        partition_counter = 0  # To keep track of partitions

        async for idx, input_data in self.do_data_loading():
            self._current_batch.append((idx, input_data))

            if len(self._current_batch) >= self.batch_size:
                batch_results = await self.do_batch_processing(
                    self._current_batch)
                results.extend(batch_results)
                self._current_batch = []

                if len(results) >= self.checkpoint_size:
                    # Convert results to DataFrame
                    df = pd.DataFrame(results, columns=['idx', 'embedding'])

                    # Save DataFrame to parquet file
                    output_file = f"{self.output_path}_{partition_counter}.parquet"
                    df.to_parquet(output_file, index=False)
                    logging.info(
                        f"Saved partition {partition_counter} to {output_file}")
                    results = []
                    partition_counter += 1

        # Process remaining items in batch
        if self._current_batch:
            batch_results = await self.do_batch_processing(self._current_batch)
            results.extend(batch_results)

        # Save final results if any
        if results:
            df = pd.DataFrame(results, columns=['idx', 'embedding'])
            partition_dir = f"{self.output_path}_part_{partition_counter}"
            os.makedirs(partition_dir, exist_ok=True)
            df.to_parquet(os.path.join(partition_dir, "data.parquet"),
                          engine='pyarrow',
                          index=False)
            logging.info(
                f"Saved final partition {partition_counter} to {partition_dir}")


async def main():
    """Example usage of the batch processing framework."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run CLIP batch processing on ImageNet')
    parser.add_argument('--output-path',
                        type=str,
                        default='embeddings.parquet',
                        help='Path to output parquet file')
    parser.add_argument('--start-idx',
                        type=int,
                        default=0,
                        help='Starting index in dataset')
    parser.add_argument('--end-idx',
                        type=int,
                        default=1000,
                        help='Ending index in dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='Batch size for processing')
    parser.add_argument('--checkpoint-size',
                        type=int,
                        default=100,
                        help='Number of results before checkpointing')
    parser.add_argument('--model-name',
                        type=str,
                        default='ViT-bigG-14',
                        help='CLIP model name')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize processor
    processor = BatchProcessor(output_path=args.output_path,
                               start_idx=args.start_idx,
                               end_idx=args.end_idx,
                               batch_size=args.batch_size,
                               checkpoint_size=args.checkpoint_size,
                               model_name=args.model_name)

    # Run processing
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())

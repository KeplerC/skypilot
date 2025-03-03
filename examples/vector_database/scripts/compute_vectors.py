"""
This script is responsible for computing the embeddings for the LAION dataset.
"""

import abc
import asyncio
import base64
from io import BytesIO
import logging
import os
from pathlib import Path
import pickle
import shutil
from typing import (Any, AsyncIterator, Dict, List, Optional, Tuple)

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
import aiohttp


class BatchProcessor():
    """Process LAION images with CLIP.
    
    This script is responsible for computing the embeddings for the LAION dataset.
    1. setup_model initializes the model
    2. get_dataset_iterator will yield individual items from the dataset
    3. do_data_loading will get an item from the dataset iterator and do any preprocessing
    4. the loaded items will be batched and handed to do_batch_processing for the ultimate processing
    """

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
        self.output_path = Path(output_path)  # Convert to Path object
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
        self.partition_counter = 0
        
        # Control parallel preprocessing
        self.preprocessing_semaphore = asyncio.Semaphore(max_preprocessing_tasks)

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
                             streaming=self.streaming,
                             trust_remote_code=True)[self.split]

        if self.start_idx > 0:
            dataset = dataset.skip(self.start_idx)

        for idx, item in enumerate(dataset, start=self.start_idx):
            if self.end_idx and idx >= self.end_idx:
                break
            yield idx, item

    async def _preprocess_input(self, url: str) -> Optional[torch.Tensor]:
        """Download and preprocess a single image from URL."""
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

    async def do_data_loading(self) -> AsyncIterator[Tuple[int, Tuple[str, torch.Tensor]]]:
        """Load and preprocess LAION images in parallel."""
        if self.model is None:
            await self.setup_model()

        preprocessing_tasks = []
        buffer_size = self.batch_size * 2

        async for idx, item in self.get_dataset_iterator():
            # Clean up completed tasks when buffer is full
            if len(preprocessing_tasks) >= buffer_size:
                done, pending = await asyncio.wait(
                    preprocessing_tasks, return_when=asyncio.FIRST_COMPLETED)
                preprocessing_tasks = list(pending)

                for task in done:
                    result = await task
                    if result[1][1] is not None:  # Check tensor in (idx, (url, tensor))
                        yield result

            # Start new preprocessing task
            async def preprocess_with_index(idx, item):
                async with self.preprocessing_semaphore:
                    url = item["url"]
                    tensor = await self._preprocess_input(url)
                    return (idx, (url, tensor))  # Return tuple with url

            task = asyncio.create_task(preprocess_with_index(idx, item))
            preprocessing_tasks.append(task)

        # Wait for and yield remaining results
        if preprocessing_tasks:
            done = await asyncio.gather(*preprocessing_tasks)
            for result in done:
                if result[1][1] is not None:  # Check tensor in (idx, (url, tensor))
                    yield result

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

        # Convert to numpy arrays
        embeddings = features.cpu().numpy()

        # Return embeddings paired with URLs
        return [(idx, pickle.dumps((url, arr)))
                for idx, url, arr in zip(indices, urls, embeddings)]

    async def find_existing_progress(self) -> Tuple[int, int]:
        """
        Find the highest processed index and partition counter from existing files.
        Returns:
            Tuple[int, int]: (highest_index, next_partition_number)
        """
        if not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            return self.start_idx, 0

        partition_files = list(
            self.output_path.parent.glob(
                f'{self.output_path.stem}_part_*.parquet'))
        print(f'Partition files: {partition_files}')
        if not partition_files:
            return self.start_idx, 0

        max_idx = self.start_idx
        max_partition = -1

        for file in partition_files:
            # Extract partition number from filename
            try:
                partition_num = int(file.stem.split('_part_')[1])
                max_partition = max(max_partition, partition_num)

                # Read the file and find highest index
                df = pd.read_parquet(file)
                if not df.empty:
                    max_idx = max(max_idx, df['idx'].max())
            except Exception as e:
                logging.warning(f'Error processing file {file}: {e}')

        return max_idx, max_partition + 1

    def save_results_to_parquet(self, results: list):
        """Save results to a parquet file with atomic write."""
        if not results:
            return

        df = pd.DataFrame(results, columns=['idx', 'output'])
        final_path = f'{self.output_path}_part_{self.partition_counter}.parquet'
        temp_path = f'/tmp/{self.partition_counter}.tmp'

        # Write to temporary file first
        df.to_parquet(temp_path, engine='pyarrow', index=False)

        # Copy from temp to final destination
        shutil.copy2(temp_path, final_path)
        os.remove(temp_path)  # Clean up temp file

        logging.info(
            f'Saved partition {self.partition_counter} to {final_path} with {len(df)} rows'
        )
        self.partition_counter += 1

    async def run(self):
        """
        Run the batch processing pipeline with recovery support.
        """
        # Initialize the model
        if self.model is None:
            await self.setup_model()

        # Find existing progress
        resume_idx, self.partition_counter = await self.find_existing_progress()
        self.start_idx = max(self.start_idx, resume_idx + 1)

        logging.info(
            f'Starting processing from index {self.start_idx} (partition {self.partition_counter})'
        )

        results = []

        async for idx, input_data in self.do_data_loading():
            self._current_batch.append((idx, input_data))
            if len(self._current_batch) >= self.batch_size:
                batch_results = await self.do_batch_processing(
                    self._current_batch)
                results.extend(batch_results)
                self._current_batch = []

                if len(results) >= self.checkpoint_size:
                    self.save_results_to_parquet(results)
                    results.clear()

        # Process any remaining items in the batch
        if self._current_batch:
            batch_results = await self.do_batch_processing(self._current_batch)
            results.extend(batch_results)

        # Write the final partition if there are any leftover results
        if results:
            self.save_results_to_parquet(results)


async def main():
    """Example usage of the batch processing framework."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run CLIP batch processing on LAION')
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
                        default=10000,
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

    parser.add_argument('--dataset-name',
                        type=str,
                        default='laion/relaion2B-en-research-safe',
                        help='LAION dataset name')

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
                               model_name=args.model_name,
                               dataset_name=args.dataset_name)

    # Run processing
    await processor.run()


if __name__ == '__main__':
    asyncio.run(main())

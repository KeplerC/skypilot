"""
This script is responsible for computing the embeddings for the LAION dataset.
"""

import abc
import asyncio
import base64
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import time
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

        # Progress tracking
        self.metrics_path = Path(output_path).parent / 'metrics'
        self.metrics_path.mkdir(exist_ok=True)
        self.worker_id = os.getenv('WORKER_ID', 'unknown')
        self.metrics_file = self.metrics_path / f'worker_{self.worker_id}.json'
        self.metrics_history_file = self.metrics_path / f'worker_{self.worker_id}_history.json'
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.session_id = f"{self.worker_id}_{int(self.start_time)}"
        
        # Load existing history if available
        self.metrics_history = self._load_metrics_history()

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
            self.failed_count += 1
            self.processed_count += 1  # Count failed items as processed
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

        self.processed_count += len(batch)
        if time.time() - self.last_update_time > 5:  # Update every 5 seconds
            self.update_metrics()
            
        return [(idx, pickle.dumps((url, arr)))
                for idx, url, arr in zip(indices, urls, embeddings)]

    async def find_existing_progress(self) -> Tuple[int, int]:
        """
        Find the highest processed index and partition counter from existing files.
        Also loads history if available to support recovery after spot VM termination.
        
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
        
        # First, load any existing history to recover from spot VM termination
        self.metrics_history = self._load_metrics_history()
        
        # If we have existing history, extract the last known position
        recovered_processed_count = 0
        if self.metrics_history:
            # Get the most recent session that was running before termination
            recent_metrics = sorted(
                [m for m in self.metrics_history if 'current_idx' in m and 'session_id' in m],
                key=lambda x: x.get('timestamp', 0)
            )
            
            # If we have history, record a termination event to mark the spot VM shutdown
            if recent_metrics and recent_metrics[-1].get('status') == 'running':
                termination_metrics = {
                    'worker_id': self.worker_id,
                    'session_id': recent_metrics[-1].get('session_id'),
                    'event': 'termination',
                    'timestamp': time.time(),
                    'processed_count': recent_metrics[-1].get('processed_count', 0),
                    'failed_count': recent_metrics[-1].get('failed_count', 0),
                    'status': 'terminated'
                }
                self.metrics_history.append(termination_metrics)
                self.save_metrics_history()  # Save this termination event
                logging.info(f"Detected spot VM termination for session {termination_metrics['session_id']}")
                
                # Recover the processed count from previous session
                recovered_processed_count = recent_metrics[-1].get('processed_count', 0)
        
        # Find highest index from partition files
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

        # Update start index based on history and partition files
        recovered_idx = max_idx
        if self.metrics_history:
            # Get the last successfully processed index from history
            for metrics in reversed(self.metrics_history):
                if 'current_idx' in metrics and metrics.get('status') != 'starting':
                    recovered_idx = max(recovered_idx, metrics.get('current_idx', self.start_idx))
                    break
        
        # Set the initial processed count to continue from previous session
        self.processed_count = recovered_processed_count
        
        logging.info(f"Recovered progress from index {recovered_idx} (partition {max_partition + 1}, processed count: {recovered_processed_count})")
        return recovered_idx, max_partition + 1

    def _load_metrics_history(self) -> List[Dict]:
        """Load existing metrics history if available."""
        try:
            if self.metrics_history_file.exists():
                with open(self.metrics_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.warning(f"Could not load metrics history: {e}")
            return []

    def save_metrics_history(self):
        """
        Save metrics history to file using atomic write.
        This is a separate function to be called at critical points for spot VM safety.
        """
        try:
            # Write history atomically
            temp_history_file = self.metrics_history_file.with_suffix('.tmp')
            with open(temp_history_file, 'w') as f:
                json.dump(self.metrics_history, f)
            shutil.copy2(temp_history_file, self.metrics_history_file)
            os.remove(temp_history_file)
        except Exception as e:
            logging.error(f"Failed to save metrics history: {e}")

    def update_metrics(self):
        """Update progress metrics file and append to history."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        metrics = {
            'worker_id': self.worker_id,
            'session_id': self.session_id,  # Add session ID to track restarts
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'current_idx': self.start_idx + self.processed_count,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'elapsed_time': elapsed_time,
            'images_per_second': self.processed_count / elapsed_time if elapsed_time > 0 else 0,
            'last_update': current_time,
            'timestamp': current_time,  # Add explicit timestamp for history tracking
            'status': 'running'
        }
        
        # Append to history
        self.metrics_history.append(metrics.copy())
        
        # Write current metrics atomically using temporary file
        temp_file = self.metrics_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(metrics, f)
        shutil.copy2(temp_file, self.metrics_file)
        os.remove(temp_file)
        
        # Save history
        self.save_metrics_history()
        
        self.last_update_time = current_time

    async def run(self):
        """
        Run the batch processing pipeline with recovery support.
        """
        try:
            # Initialize the model
            if self.model is None:
                await self.setup_model()

            # Find existing progress and recover state
            resume_idx, self.partition_counter = await self.find_existing_progress()
            self.start_idx = max(self.start_idx, resume_idx + 1)

            logging.info(
                f'Starting processing from index {self.start_idx} (partition {self.partition_counter})'
            )
            
            # Record start event in history
            start_metrics = {
                'worker_id': self.worker_id,
                'session_id': self.session_id,
                'event': 'start',
                'start_idx': self.start_idx,
                'end_idx': self.end_idx,
                'timestamp': time.time(),
                'status': 'starting'
            }
            self.metrics_history.append(start_metrics)
            self.save_metrics_history()  # Explicitly save history for recovery
            self.update_metrics()  # Also save current state

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
                        # Save metrics history at each checkpoint for recovery
                        self.update_metrics()

            # Process any remaining items in the batch
            if self._current_batch:
                batch_results = await self.do_batch_processing(self._current_batch)
                results.extend(batch_results)

            # Write the final partition if there are any leftover results
            if results:
                self.save_results_to_parquet(results)

            # Write final metrics
            self.update_metrics()
            
            # Update status to completed
            completion_metrics = {
                'worker_id': self.worker_id,
                'session_id': self.session_id,
                'event': 'completion',
                'processed_count': self.processed_count,
                'failed_count': self.failed_count,
                'timestamp': time.time(),
                'status': 'completed'
            }
            self.metrics_history.append(completion_metrics)
            
            # Update current status file
            with open(self.metrics_file, 'r') as f:
                metrics = json.loads(f.read())
            metrics['status'] = 'completed'
            
            # Write atomically
            temp_file = self.metrics_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metrics, f)
            shutil.copy2(temp_file, self.metrics_file)
            os.remove(temp_file)
            
            # Save final history
            self.save_metrics_history()
            
        except Exception as e:
            # Record error event
            error_metrics = {
                'worker_id': self.worker_id,
                'session_id': self.session_id,
                'event': 'error',
                'error': str(e),
                'timestamp': time.time(),
                'status': 'failed'
            }
            self.metrics_history.append(error_metrics)
            
            # Update status to failed in current metrics
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics = json.loads(f.read())
                metrics['status'] = 'failed'
                metrics['error'] = str(e)
                
                # Write atomically
                temp_file = self.metrics_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(metrics, f)
                shutil.copy2(temp_file, self.metrics_file)
                os.remove(temp_file)
                
                # Save history with error
                self.save_metrics_history()
            except Exception as nested_e:
                logging.error(f"Failed to update error status: {nested_e}")
            
            raise

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

import argparse
import asyncio
from asyncio import Semaphore
import base64
import csv
from io import BytesIO
import os
from typing import Dict, List, Optional, Tuple

import aiohttp
from datasets import load_dataset
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from tqdm import tqdm
import logging

# Add logging setup near the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AsyncDataLoader:

    def __init__(self,
                 start_idx: int,
                 end_idx: int,
                 inference_server_url: str = "http://localhost:8000",
                 output_path: str = "embeddings.parquet",
                 max_concurrent: int = 50):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.inference_server_url = inference_server_url
        self.output_path = output_path
        self.csv_path = output_path.replace('.parquet', '_intermediate.csv')
        self.semaphore = Semaphore(max_concurrent)
        self.session = None
        # Store a small batch of results in memory before writing to CSV
        self.results: List[Dict] = []
        self.batch_size = 100  # Write to CSV every 100 items

        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['idx', 'url', 'embedding'])

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_image(self, url: str) -> Optional[str]:
        """Download image from URL with retry mechanism"""
        try:
            async with self.semaphore:  # Limit concurrent connections
                async with self.session.get(
                        url, timeout=10,
                        ssl=False) as response:  # Disable SSL verification
                    if response.status == 200:
                        data = await response.read()
                        return base64.b64encode(data).decode()
        except Exception as e:
            logging.debug(f"Error downloading image from {url}: {str(e)}")
        return None

    async def get_embedding(self, image_data: str,
                            idx: int) -> Optional[np.ndarray]:
        """Get CLIP embedding for a single image"""
        try:
            async with self.session.post(
                    f"{self.inference_server_url}/v1/embeddings",
                    json={
                        "input": image_data,
                        "model": "ViT-bigG-14"
                    }) as response:
                result = await response.json()
                if result["data"]:
                    return np.array(result["data"][0]["embedding"])
        except Exception as e:
            logging.debug(f"Error getting embedding for index {idx}: {str(e)}")
        return None

    def create_dataset(self):
        """Create a dataset iterator"""
        dataset = load_dataset("laion/relaion2B-en-research-safe",
                               streaming=True)["train"]

        if self.start_idx > 0:
            dataset = dataset.skip(self.start_idx)
        dataset = dataset.take(self.end_idx - self.start_idx)
        return dataset

    def append_to_csv(self):
        """Append current results to CSV file"""
        if not self.results:
            return

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for result in self.results:
                writer.writerow([
                    result['idx'],
                    result['url'],
                    ','.join(map(str, result['embedding']))
                ])
        
        self.results = []  # Clear memory after writing

    def convert_to_parquet(self):
        """Convert the final CSV to parquet format"""
        if not os.path.exists(self.csv_path):
            logging.info("No results to convert")
            return

        # Read CSV in chunks to handle large files
        chunks = pd.read_csv(self.csv_path, 
                           chunksize=10000,
                           names=['idx', 'url', 'embedding'],
                           skiprows=1)
        
        # Process first chunk
        first_chunk = next(chunks)
        first_chunk['embedding'] = first_chunk['embedding'].apply(
            lambda x: [float(i) for i in x.split(',')])
        
        # Write first chunk to parquet
        table = pa.Table.from_pandas(first_chunk)
        pq.write_table(table, self.output_path)
        
        # Append remaining chunks
        for chunk in chunks:
            chunk['embedding'] = chunk['embedding'].apply(
                lambda x: [float(i) for i in x.split(',')])
            table = pa.Table.from_pandas(chunk)
            pq.write_table(table, self.output_path, append=True)
        
        # Clean up intermediate CSV
        os.remove(self.csv_path)
        logging.info(f"Successfully converted results to {self.output_path}")

    async def process_single_item(self, item, idx: int) -> bool:
        """Process a single item from the dataset"""
        try:
            # Download image first
            image_data = await self.download_image(item["url"])
            if image_data is None:
                return False

            # Get embedding in a separate step
            embedding = await self.get_embedding(image_data, idx)
            if embedding is None:
                return False

            # Store result
            self.results.append({
                "idx": idx,
                "url": item["url"],
                "embedding": embedding.tolist()
            })

            # Write to CSV if batch size is reached
            if len(self.results) >= self.batch_size:
                self.append_to_csv()

            return True
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            return False

    async def run(self):
        dataset = self.create_dataset()
        total = self.end_idx - self.start_idx

        # Create queues for download and embedding tasks
        download_queue = asyncio.Queue(maxsize=500)  # Buffer downloaded images
        embedding_queue = asyncio.Queue(maxsize=500)  # Buffer images ready for embedding
        
        # Create progress bars for each stage
        with tqdm(total=total, desc="Total Progress") as total_pbar, \
             tqdm(total=total, desc="Downloads", position=1) as download_pbar, \
             tqdm(total=total, desc="Embeddings", position=2) as embedding_pbar:

            async def download_worker():
                """Worker to download images and put them in embedding queue"""
                try:
                    while True:
                        idx, item = await download_queue.get()
                        try:
                            image_data = await self.download_image(item["url"])
                            if image_data is not None:
                                await embedding_queue.put((idx, item["url"], image_data))
                            download_pbar.update(1)
                        except Exception as e:
                            logging.error(f"Error downloading image {idx}: {str(e)}")
                        finally:
                            download_queue.task_done()
                except asyncio.CancelledError:
                    return

            async def embedding_worker():
                """Worker to process embeddings from the embedding queue"""
                try:
                    while True:
                        idx, url, image_data = await embedding_queue.get()
                        try:
                            embedding = await self.get_embedding(image_data, idx)
                            if embedding is not None:
                                self.results.append({
                                    "idx": idx,
                                    "url": url,
                                    "embedding": embedding.tolist()
                                })
                                if len(self.results) >= self.batch_size:
                                    self.append_to_csv()
                            embedding_pbar.update(1)
                            total_pbar.update(1)
                        except Exception as e:
                            logging.error(f"Error processing embedding {idx}: {str(e)}")
                        finally:
                            embedding_queue.task_done()
                except asyncio.CancelledError:
                    return

            # Start workers
            download_workers = [
                asyncio.create_task(download_worker())
                for _ in range(50)  # Number of concurrent downloads
            ]
            embedding_workers = [
                asyncio.create_task(embedding_worker())
                for _ in range(10)  # Number of concurrent embedding processes
            ]

            # Feed the download queue
            for idx, item in enumerate(dataset, start=self.start_idx):
                await download_queue.put((idx, item))

            # Wait for all downloads to complete
            await download_queue.join()
            
            # Wait for all embeddings to complete
            await embedding_queue.join()

            # Cancel workers
            for worker in download_workers + embedding_workers:
                worker.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*download_workers, *embedding_workers, 
                               return_exceptions=True)

        # Write any remaining results to CSV
        if self.results:
            self.append_to_csv()

        # Convert final CSV to parquet
        self.convert_to_parquet()


async def main():
    parser = argparse.ArgumentParser(
        description="Process images and generate CLIP embeddings")
    parser.add_argument("--start-idx",
                        type=int,
                        required=True,
                        help="Starting index in the dataset")
    parser.add_argument("--end-idx",
                        type=int,
                        required=True,
                        help="Ending index in the dataset")
    parser.add_argument("--server-url",
                        type=str,
                        default="http://localhost:5005",
                        help="Inference server URL")
    parser.add_argument("--output",
                        type=str,
                        default="embeddings.parquet",
                        help="Output parquet file path")

    args = parser.parse_args()

    async with AsyncDataLoader(start_idx=args.start_idx,
                               end_idx=args.end_idx,
                               inference_server_url=args.server_url,
                               output_path=args.output) as loader:
        await loader.run()


if __name__ == "__main__":
    asyncio.run(main())

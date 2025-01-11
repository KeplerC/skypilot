import argparse
import asyncio
import base64
from io import BytesIO
import os
from typing import List, Optional, Tuple, Dict

import aiohttp
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

class AsyncDataLoader:
    def __init__(self,
                start_idx: int,
                end_idx: int,
                inference_server_url: str = "http://localhost:8000",
                output_path: str = "embeddings.parquet"):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.inference_server_url = inference_server_url
        self.output_path = output_path
        self.session = None
        # Store results in memory
        self.results: List[Dict] = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def download_image(self, url: str) -> Optional[str]:
        """Download image from URL and return raw bytes as base64 string"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.read()
                    return base64.b64encode(data).decode()
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
        return None

    async def get_embedding(self, image_data: str, idx: int) -> Optional[np.ndarray]:
        """Get CLIP embedding for a single image"""
        try:
            async with self.session.post(
                f"{self.inference_server_url}/v1/embeddings",
                json={"input": image_data, "model": "ViT-B/32"}
            ) as response:
                result = await response.json()
                if result["data"]:
                    return np.array(result["data"][0]["embedding"])
        except Exception as e:
            print(f"Error getting embedding for index {idx}: {str(e)}")
        return None

    def create_dataset(self):
        """Create a dataset iterator"""
        dataset = load_dataset(
            "laion/relaion2B-en-research-safe",
            streaming=True
        )["train"]

        if self.start_idx > 0:
            dataset = dataset.skip(self.start_idx)
        dataset = dataset.take(self.end_idx - self.start_idx)
        return dataset

    def save_results(self):
        """Save all collected results to a single parquet file"""
        if not self.results:
            print("No results to save")
            return

        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.output_path)
        print(f"Saved {len(self.results)} embeddings to {self.output_path}")

    async def process_single_item(self, item, idx: int) -> bool:
        """Process a single item from the dataset"""
        image_data = await self.download_image(item["url"])
        if image_data is None:
            return False

        embedding = await self.get_embedding(image_data, idx)
        if embedding is None:
            return False

        # Store result in memory
        self.results.append({
            "idx": idx,
            "url": item["url"],
            "embedding": embedding.tolist()
        })
        return True

    async def run(self):
        dataset = self.create_dataset()
        total = self.end_idx - self.start_idx
        
        # Process items concurrently in chunks to control memory usage
        chunk_size = 100  # Adjust based on memory constraints
        tasks = []
        
        with tqdm(total=total, desc="Processing images") as pbar:
            for idx, item in enumerate(dataset, start=self.start_idx):
                task = asyncio.create_task(self.process_single_item(item, idx))
                tasks.append(task)
                
                # When chunk is full or at the end, wait for completion
                if len(tasks) >= chunk_size or idx == self.end_idx - 1:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, bool):
                            pbar.set_postfix({"status": "success" if result else "failed"})
                        else:
                            pbar.set_postfix({"status": "error"})
                        pbar.update(1)
                    tasks = []
        
        # Save all results at once
        self.save_results()

async def main():
    parser = argparse.ArgumentParser(description="Process images and generate CLIP embeddings")
    parser.add_argument("--start-idx", type=int, required=True, help="Starting index in the dataset")
    parser.add_argument("--end-idx", type=int, required=True, help="Ending index in the dataset")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="Inference server URL")
    parser.add_argument("--output", type=str, default="embeddings.parquet", help="Output parquet file path")
    
    args = parser.parse_args()
    
    async with AsyncDataLoader(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        inference_server_url=args.server_url,
        output_path=args.output
    ) as loader:
        await loader.run()

if __name__ == "__main__":
    asyncio.run(main())

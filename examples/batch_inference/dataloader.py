import asyncio
import base64
from io import BytesIO
import os
from typing import List, Optional, Tuple

import aiohttp
from datasets import load_dataset
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


class AsyncDataLoader:

    def __init__(self,
                 batch_size=32,
                 start_idx=0,
                 end_idx=None,
                 inference_server_url="http://localhost:8000",
                 output_file="embeddings.parquet"):
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.inference_server_url = inference_server_url
        self.output_file = output_file
        self.session = None

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

    async def process_batch(self, batch) -> Tuple[List[str], List[int]]:
        """Process a batch of URLs and return valid image data"""
        tasks = [self.download_image(url) for url in batch['url']]
        image_data = await asyncio.gather(*tasks)

        valid_images = []
        valid_indices = []
        for idx, img_data in enumerate(image_data):
            if img_data is not None:
                valid_images.append(img_data)
                valid_indices.append(idx)

        return valid_images, valid_indices

    async def get_embeddings(
            self, image_data: List[str],
            batch_id: str) -> Tuple[List[np.ndarray], List[int]]:
        """Get CLIP embeddings for a batch of images"""
        # Send request to inference server with raw image data
        async with self.session.post(f"{self.inference_server_url}/embed",
                                     json={
                                         "batch_id": batch_id,
                                         "images": image_data
                                     }) as response:
            result = await response.json()
            return [np.array(emb) for emb in result["embeddings"]
                   ], result["valid_indices"]

    def create_dataset(self):
        """Create a dataset iterator"""
        dataset = load_dataset("laion/relaion2B-en-research-safe",
                               streaming=True)['train']

        if self.start_idx > 0:
            dataset = dataset.skip(self.start_idx)
        if self.end_idx is not None:
            dataset = dataset.take(self.end_idx - self.start_idx)

        return dataset.iter(batch_size=self.batch_size)

    async def save_embeddings(self, urls: List[str],
                              embeddings: List[np.ndarray]):
        """Save embeddings to parquet file with append mode"""
        # Convert embeddings to lists for storage
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Create DataFrame
        df = pd.DataFrame({'url': urls, 'embedding': embeddings_list})

        # Convert to Arrow table
        table = pa.Table.from_pandas(df)

        # Write to parquet file (append if exists)
        if os.path.exists(self.output_file):
            pq.write_to_dataset(table,
                                self.output_file,
                                filesystem=pa.LocalFileSystem(),
                                existing_data_behavior='append_to_dataset')
        else:
            pq.write_table(table, self.output_file)


async def main():
    async with AsyncDataLoader(batch_size=32) as loader:
        dataset = loader.create_dataset()

        for batch_idx, batch in enumerate(dataset):
            # Download and process images from URLs
            images, download_indices = await loader.process_batch(batch)

            if images:
                # Get CLIP embeddings for valid images
                embeddings, embed_indices = await loader.get_embeddings(
                    images, f"batch_{batch_idx}")

                # Map back to original indices
                valid_indices = [download_indices[i] for i in embed_indices]
                valid_urls = [batch['url'][i] for i in valid_indices]

                print(
                    f"Batch {batch_idx}: {len(valid_urls)} successful out of {len(batch['url'])} total"
                )

                # Save embeddings to parquet file
                await loader.save_embeddings(valid_urls, embeddings)


if __name__ == "__main__":
    asyncio.run(main())

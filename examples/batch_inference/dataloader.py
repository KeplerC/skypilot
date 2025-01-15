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

# Get node rank and total nodes from environment variables
NODE_RANK = int(os.getenv('SKYPILOT_NODE_RANK', '0'))
NUM_NODES = int(os.getenv('SKYPILOT_NUM_NODES', '1'))

def calculate_node_range(start_idx: int, end_idx: int, node_rank: int, num_nodes: int) -> Tuple[int, int]:
    """Calculate the range of indices this node should process.
    
    Args:
        start_idx: Global start index
        end_idx: Global end index
        node_rank: Current node's rank (0-based)
        num_nodes: Total number of nodes
        
    Returns:
        Tuple of (node_start_idx, node_end_idx)
    """
    total_range = end_idx - start_idx
    chunk_size = total_range // num_nodes
    remainder = total_range % num_nodes
    
    # Distribute remainder across first few nodes
    node_start = start_idx + (node_rank * chunk_size) + min(node_rank, remainder)
    if node_rank < remainder:
        chunk_size += 1
    node_end = node_start + chunk_size
    
    return node_start, node_end

class AsyncDataLoader:

    def __init__(self,
                 start_idx: int,
                 end_idx: int,
                 inference_server_url: str = "http://localhost:8000",
                 output_path: str = "embeddings.parquet",
                 csv_path: str = "/tmp/intermediate.csv",
                 max_concurrent: int = 50):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.inference_server_url = inference_server_url
        self.output_path = output_path
        self.csv_path = csv_path
        self.semaphore = Semaphore(max_concurrent)
        self.session = None
        # Store a small batch of results in memory before writing to CSV
        self.results: List[Dict] = []
        self.batch_size = 50  # Write to CSV every 100 items

        # Create CSV file with headers if it doesn't exist
        try:
            self.results = pd.read_csv(self.csv_path).to_dict(orient='records')
            self.start_idx = list(self.results[-1].values())[0]
            logging.info(f"Resuming from index {self.start_idx} based on existing output")
        except Exception as e:
            logging.info(f"No existing output found, starting from index {self.start_idx}")
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

        # Read existing content
        existing_rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', newline='') as f:
                existing_rows = list(csv.reader(f))
        
        # Write all content
        with open(self.csv_path, 'w', newline='') as f:
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

    async def get_embeddings_batch(self, batch_data: List[Tuple[int, str, str]]) -> List[Tuple[int, str, Optional[np.ndarray]]]:
        """Get CLIP embeddings for a batch of images"""
        try:
            # Prepare batch input
            inputs = [{"input": image_data, "model": "ViT-bigG-14"} for _, _, image_data in batch_data]
            
            async with self.session.post(
                    f"{self.inference_server_url}/v1/embeddings/batch",
                    json={"inputs": inputs}) as response:
                result = await response.json()
                
                # Process results maintaining order
                processed_results = []
                for (idx, url, _), embedding_result in zip(batch_data, result.get("data", [])):
                    embedding = np.array(embedding_result["embedding"]) if embedding_result else None
                    processed_results.append((idx, url, embedding))
                return processed_results
        except Exception as e:
            logging.error(f"Error getting batch embeddings: {str(e)}")
            # Return None embeddings for all items in batch
            return [(idx, url, None) for idx, url, _ in batch_data]

    async def run(self):
        dataset = self.create_dataset()
        total = self.end_idx - self.start_idx

        # Create queues for download, embedding, and writing tasks
        download_queue = asyncio.Queue(maxsize=500)
        embedding_queue = asyncio.Queue(maxsize=500)
        write_queue = asyncio.Queue(maxsize=500)  # New queue for writing results
        
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
                """Worker to process embeddings from the embedding queue in batches"""
                try:
                    while True:
                        # Collect batch of items
                        batch = []
                        batch_size = self.batch_size
                        
                        try:
                            item = await embedding_queue.get()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            continue

                        while len(batch) < batch_size:
                            try:
                                item = embedding_queue.get_nowait()
                                batch.append(item)
                            except asyncio.QueueEmpty:
                                break
                        
                        try:
                            results = await self.get_embeddings_batch(batch)
                            for idx, url, embedding in results:
                                if embedding is not None:
                                    # Instead of storing in self.results, send to write queue
                                    await write_queue.put({
                                        "idx": idx,
                                        "url": url,
                                        "embedding": embedding.tolist()
                                    })
                                embedding_pbar.update(1)
                                total_pbar.update(1)
                        except Exception as e:
                            logging.error(f"Error processing batch: {str(e)}")
                        finally:
                            for _ in batch:
                                embedding_queue.task_done()
                except asyncio.CancelledError:
                    return

            async def csv_writer():
                """Dedicated worker for writing results to CSV"""
                try:
                    results_buffer = []
                    while True:
                        try:
                            result = await write_queue.get()
                            results_buffer.append(result)
                            
                            if len(results_buffer) >= self.batch_size:
                                # Read existing content
                                existing_rows = []
                                if os.path.exists(self.csv_path):
                                    with open(self.csv_path, 'r', newline='') as f:
                                        reader = csv.reader(f)
                                        existing_rows = list(reader)
                                
                                # Write all content
                                with open(self.csv_path, 'w', newline='') as f:
                                    writer = csv.writer(f)
                                    # Write header if it's a new file
                                    if not existing_rows:
                                        writer.writerow(['idx', 'url', 'embedding'])
                                    else:
                                        writer.writerows(existing_rows[1:])  # Skip header
                                    # Write new results
                                    for r in results_buffer:
                                        writer.writerow([
                                            r['idx'],
                                            r['url'],
                                            ','.join(map(str, r['embedding']))
                                        ])
                                results_buffer = []
                        except Exception as e:
                            logging.error(f"Error writing to CSV: {str(e)}")
                        finally:
                            write_queue.task_done()
                except asyncio.CancelledError:
                    # Write any remaining results before exiting
                    if results_buffer:
                        existing_rows = []
                        if os.path.exists(self.csv_path):
                            with open(self.csv_path, 'r', newline='') as f:
                                reader = csv.reader(f)
                                existing_rows = list(reader)
                        
                        with open(self.csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            if not existing_rows:
                                writer.writerow(['idx', 'url', 'embedding'])
                            else:
                                writer.writerows(existing_rows[1:])  # Skip header
                            for r in results_buffer:
                                writer.writerow([
                                    r['idx'],
                                    r['url'],
                                    ','.join(map(str, r['embedding']))
                                ])
                    return

            # Start workers
            download_workers = [
                asyncio.create_task(download_worker())
                for _ in range(100)
            ]
            embedding_workers = [
                asyncio.create_task(embedding_worker())
                for _ in range(50)
            ]
            csv_writer_task = asyncio.create_task(csv_writer())  # Single CSV writer

            # Feed the download queue
            for idx, item in enumerate(dataset, start=self.start_idx):
                await download_queue.put((idx, item))

            # Wait for all queues to complete
            await download_queue.join()
            await embedding_queue.join()
            await write_queue.join()

            # Cancel all workers
            for worker in download_workers + embedding_workers + [csv_writer_task]:
                worker.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*download_workers, *embedding_workers, csv_writer_task,
                               return_exceptions=True)

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
    parser.add_argument("--csv-path",
                        type=str,
                        default="intermediate.csv",
                        help="Intermediate CSV file path")

    args = parser.parse_args()

    # Calculate node-specific range
    node_start, node_end = calculate_node_range(
        args.start_idx, args.end_idx, NODE_RANK, NUM_NODES
    )
    
    # Modify output path to include node rank
    output_path = args.output
    if NUM_NODES > 1:
        base, ext = os.path.splitext(args.output)
        output_path = f"{base}_node{NODE_RANK}{ext}"
    
    csv_path = args.csv_path
    if NUM_NODES > 1:
        base, ext = os.path.splitext(csv_path)
        csv_path = f"{base}_node{NODE_RANK}{ext}"
    
    logging.info(f"Node {NODE_RANK}/{NUM_NODES} processing range [{node_start}, {node_end})")
    logging.info(f"Output will be saved to: {output_path}")

    

    async with AsyncDataLoader(start_idx=node_start,
                             end_idx=node_end,
                             inference_server_url=args.server_url,
                             output_path=output_path,
                             csv_path=csv_path) as loader:
        await loader.run()


if __name__ == "__main__":
    asyncio.run(main())

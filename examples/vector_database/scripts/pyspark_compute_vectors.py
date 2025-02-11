"""
PySpark implementation of the vector computation pipeline.
This script uses Spark for distributed processing and CLIP for computing embeddings.
"""

import argparse
import os
from pathlib import Path
import pickle
from typing import Iterator, Tuple, List

import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType
import torch
from tqdm import tqdm


def create_spark_session(app_name: str) -> SparkSession:
    """Create a Spark session with appropriate configuration."""
    spark = (SparkSession.builder
            .appName(app_name)
            .config('spark.task.maxFailures', 10)
            .config('spark.executor.heartbeatInterval', '20s')
            .config('spark.network.timeout', '800s')
            # GPU resource configurations
            .config('spark.executor.resource.gpu.amount', 1)  # Each executor gets 1 GPU
            .config('spark.executor.resource.gpu.discoveryScript', '/usr/bin/nvidia-smi')  # Path to nvidia-smi
            .config('spark.executor.resource.gpu.vendor', 'nvidia.com')
            .config('spark.task.resource.gpu.amount', 1)  # Each task gets 1 GPU
            .getOrCreate())
    
    # Set checkpoint directory
    spark.sparkContext.setCheckpointDir('/tmp/spark_checkpoints')
    return spark


class CLIPProcessor:
    """Wrapper for CLIP model to compute embeddings."""
    
    def __init__(self, model_name: str = 'ViT-bigG-14',
                 pretrained: str = 'laion2b_s39b_b160k',
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()

    def compute_embedding(self, image: Image.Image) -> np.ndarray:
        """Compute embedding for a single image."""
        if image is None:
            return None
        
        try:
            # Preprocess and compute embedding
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()[0]
        except Exception as e:
            print(f'Error computing embedding: {e}')
            return None


def process_partition(iterator: Iterator[Tuple[int, dict]],
                     model_name: str,
                     pretrained: str) -> Iterator[Tuple[int, bytes]]:
    """Process a partition of the dataset.
    
    Args:
        iterator: Iterator over (index, item) pairs from the dataset
        model_name: CLIP model name
        pretrained: CLIP model pretrained weights
    
    Yields:
        Tuples of (index, pickled embedding)
    """
    # Initialize CLIP processor
    processor = CLIPProcessor(model_name=model_name, pretrained=pretrained)
    
    # Process items in batches for better GPU utilization
    batch = []
    batch_size = 32  # Process 32 images at a time
    
    for idx, item in iterator:
        try:
            image = item['image']
            if image is None:
                continue
            
            batch.append((idx, image))
            
            if len(batch) >= batch_size:
                yield from process_batch(batch, processor)
                batch = []
                
        except Exception as e:
            print(f'Error processing index {idx}: {e}')
            continue
    
    # Process remaining items
    if batch:
        yield from process_batch(batch, processor)


def process_batch(batch: List[Tuple[int, Image.Image]], 
                 processor: CLIPProcessor) -> Iterator[Tuple[int, bytes]]:
    """Process a batch of images together."""
    for idx, image in batch:
        embedding = processor.compute_embedding(image)
        if embedding is not None:
            yield (idx, pickle.dumps(embedding))


def create_dataset_rdd(spark, start_idx: int, end_idx: int, num_partitions: int):
    from datasets import load_dataset
    
    # Load dataset once
    dataset = load_dataset('ILSVRC/imagenet-1k',
                          streaming=True,
                          trust_remote_code=True)['train']
    
    # Create batches of indices
    batch_size = (end_idx - start_idx) // num_partitions
    
    def process_batch(start, end):
        items = list(dataset.skip(start).take(end - start))
        return [(idx, item) for idx, item in enumerate(items, start=start)]
    
    # Create RDD with batched loading
    return spark.sparkContext.parallelize(
        [(i, i + batch_size) for i in range(start_idx, end_idx, batch_size)],
        num_partitions
    ).flatMap(lambda x: process_batch(*x))

def load_dataset_item(idx: int):
    """Load a single item from the dataset."""
    from datasets import load_dataset
    
    # Load dataset in streaming mode
    dataset = load_dataset('ILSVRC/imagenet-1k',
                          streaming=True,
                          trust_remote_code=True)['train']
    
    # Skip to the desired index and take one item
    return next(iter(dataset.skip(idx).take(1)))


def main():
    parser = argparse.ArgumentParser(
        description='Distributed CLIP inference using PySpark')
    parser.add_argument('--output-path',
                       type=str,
                       default='/data/embeddings',
                       help='Path to output directory')
    parser.add_argument('--start-idx',
                       type=int,
                       default=0,
                       help='Starting index in dataset')
    parser.add_argument('--end-idx',
                       type=int,
                       default=10000,
                       help='Ending index in dataset')
    parser.add_argument('--num-partitions',
                       type=int,
                       default=100,
                       help='Number of Spark partitions')
    parser.add_argument('--model-name',
                       type=str,
                       default='ViT-bigG-14',
                       help='CLIP model name')
    parser.add_argument('--pretrained',
                       type=str,
                       default='laion2b_s39b_b160k',
                       help='CLIP pretrained weights')
    args = parser.parse_args()

    # Initialize Spark with GPU configuration
    spark = create_spark_session('CLIP Vector Computation')

    # Create dataset RDD with proper partitioning
    rdd = create_dataset_rdd(spark, args.start_idx, args.end_idx, args.num_partitions)
    
    # Define schema for the output DataFrame
    schema = StructType([
        StructField('idx', IntegerType(), False),
        StructField('embedding', BinaryType(), False)
    ])
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('/tmp/spark_checkpoints', exist_ok=True)
    
    # Add monitoring
    processed_rdd = rdd.mapPartitions(
        lambda iterator: process_partition(
            iterator,
            args.model_name,
            args.pretrained
        )).cache()  # Cache to prevent recomputation
    
    # Add progress monitoring
    count = processed_rdd.count()
    print(f"Processed {count} items")
    
    # Add periodic checkpointing
    if count > 0:
        processed_rdd.checkpoint()
    
    # Convert to DataFrame and write in parallel
    df = spark.createDataFrame(processed_rdd, schema)
    
    # Write output in parallel using partitioning
    df.write.parquet(
        args.output_path,
        mode='append',
        partitionBy=['idx']
    )

    spark.stop()


if __name__ == '__main__':
    main() 
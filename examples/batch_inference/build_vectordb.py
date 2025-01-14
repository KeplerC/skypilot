import os
import boto3
import pandas as pd
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import argparse
import logging
from io import BytesIO
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_s3_parquet_files(bucket_name: str, prefix: str):
    """List all parquet files in the specified S3 bucket and prefix."""
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    parquet_files = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.parquet'):
                    parquet_files.append(obj['Key'])
    
    return parquet_files

def read_parquet_from_s3(bucket_name: str, key: str) -> pd.DataFrame:
    """Read a parquet file from S3 into a pandas DataFrame."""
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    
    # Read the content into a BytesIO buffer first
    buffer = BytesIO(response['Body'].read())
    return pd.read_parquet(buffer)

def process_batch(client, collection, batch_df):
    """Process a batch of data and add it to the ChromaDB collection."""
    # Extract data from DataFrame
    ids = [str(idx) for idx in batch_df['idx']]
    embeddings = batch_df['embedding'].tolist()
    metadatas = [{'url': url} for url in batch_df['url']]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

def main():
    parser = argparse.ArgumentParser(description='Build ChromaDB from S3 parquet files')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, required=True, help='S3 prefix/folder containing parquet files')
    parser.add_argument('--collection-name', type=str, default='clip_embeddings', help='ChromaDB collection name')
    parser.add_argument('--persist-dir', type=str, default='./chroma_db', help='Directory to persist ChromaDB')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    if os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir)
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=args.persist_dir)
    
    # Create or get collection
    try:
        collection = client.create_collection(
            name=args.collection_name,
            metadata={"description": "CLIP embeddings from LAION dataset"}
        )
        logger.info(f"Created new collection: {args.collection_name}")
    except ValueError:
        collection = client.get_collection(name=args.collection_name)
        logger.info(f"Using existing collection: {args.collection_name}")
    
    # List parquet files
    parquet_files = list_s3_parquet_files(args.bucket, args.prefix)
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Process each parquet file
    for parquet_file in parquet_files:
        logger.info(f"Processing {parquet_file}")
        df = read_parquet_from_s3(args.bucket, parquet_file)
        
        # Process in batches
        for i in tqdm(range(0, len(df), args.batch_size)):
            batch_df = df.iloc[i:i + args.batch_size]
            process_batch(client, collection, batch_df)
    
    logger.info("Vector database build complete!")
    logger.info(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main() 
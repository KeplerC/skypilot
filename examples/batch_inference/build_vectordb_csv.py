import os
import boto3
import pandas as pd
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import argparse
import logging
from io import BytesIO, StringIO
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_s3_csv_files(bucket_name: str, prefix: str):
    """List all intermediate CSV files in the specified S3 bucket and prefix."""
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    csv_files = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.csv') and 'intermediate' in obj['Key']:
                    csv_files.append(obj['Key'])
    
    return sorted(csv_files)  # Sort to process in order

def read_csv_from_s3(bucket_name: str, key: str, chunk_size: int) -> pd.DataFrame:
    """Read a CSV file from S3 in chunks."""
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    
    # Read the content into a StringIO buffer
    csv_string = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string), 
                      names=['idx', 'url', 'embedding'],
                      skiprows=1,
                      chunksize=chunk_size)

def process_batch(collection, batch_df):
    """Process a batch of data and add it to the ChromaDB collection."""
    # Convert embedding strings to lists of floats
    embeddings = batch_df['embedding'].apply(
        lambda x: [float(i) for i in x.split(',')]
    ).tolist()
    
    # Extract data from DataFrame
    ids = [str(idx) for idx in batch_df['idx']]
    metadatas = [{'url': url} for url in batch_df['url']]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

def main():
    parser = argparse.ArgumentParser(description='Build ChromaDB from S3 CSV files')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, required=True, help='S3 prefix/folder containing CSV files')
    parser.add_argument('--collection-name', type=str, default='clip_embeddings', help='ChromaDB collection name')
    parser.add_argument('--persist-dir', type=str, default='./chroma_db', help='Directory to persist ChromaDB')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()

    # Clear existing database if it exists
    if os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=args.persist_dir)
    
    # Create collection
    collection = client.create_collection(
        name=args.collection_name,
        metadata={"description": "CLIP embeddings from LAION dataset"}
    )
    logger.info(f"Created new collection: {args.collection_name}")
    
    # List CSV files
    csv_files = list_s3_csv_files(args.bucket, args.prefix)
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file}")
        
        # Process file in chunks
        chunks = read_csv_from_s3(args.bucket, csv_file, args.batch_size)
        for chunk in tqdm(chunks, desc=f"Processing {os.path.basename(csv_file)}"):
            process_batch(collection, chunk)
    
    logger.info("Vector database build complete!")
    logger.info(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main() 
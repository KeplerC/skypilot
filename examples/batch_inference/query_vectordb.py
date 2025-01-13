import argparse
import chromadb
import numpy as np
import open_clip
import torch
from typing import List, Tuple


def encode_text(text: str, model_name: str = "ViT-bigG-14") -> np.ndarray:
    """Encode text using CLIP model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained="laion2b_s39b_b160k",
        device=device
    )
    
    # Tokenize and encode
    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer([text]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize the features
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()


def query_collection(collection, query_embedding: np.ndarray, n_results: int = 5) -> List[Tuple[str, float]]:
    """Query the collection and return top matches with scores."""
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    
    # Combine URLs and distances
    urls = [item['url'] for item in results['metadatas'][0]]
    distances = results['distances'][0]
    
    # Convert distances to similarities (cosine similarity = 1 - distance/2)
    similarities = [1 - (d / 2) for d in distances]
    
    return list(zip(urls, similarities))


def main():
    parser = argparse.ArgumentParser(description='Query ChromaDB with text input')
    parser.add_argument('--text', type=str, required=True, help='Text query to search for')
    parser.add_argument('--collection-name', type=str, default='clip_embeddings', 
                       help='ChromaDB collection name')
    parser.add_argument('--persist-dir', type=str, default='./chroma_db', 
                       help='Directory where ChromaDB is persisted')
    parser.add_argument('--n-results', type=int, default=5, 
                       help='Number of results to return')
    parser.add_argument('--model', type=str, default='ViT-bigG-14',
                       help='CLIP model to use for text encoding')
    
    args = parser.parse_args()
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=args.persist_dir)
    
    try:
        # Get the collection
        collection = client.get_collection(name=args.collection_name)
        print(f"\nQuerying collection: {args.collection_name}")
        print(f"Total documents in collection: {collection.count()}")
        
        # Encode the query text
        print(f"\nEncoding query: '{args.text}'")
        query_embedding = encode_text(args.text, args.model)
        
        # Query the collection
        results = query_collection(collection, query_embedding, args.n_results)
        
        # Print results
        print(f"\nTop {args.n_results} matches:")
        print("-" * 80)
        for url, similarity in results:
            print(f"Similarity: {similarity:.4f}")
            print(f"URL: {url}")
            print("-" * 80)
            
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Make sure the collection exists and the persist_dir is correct.")


if __name__ == "__main__":
    main() 
import asyncio
import logging

from processors import ClipBatchProcessor

async def main():
    """Example usage of the batch processing framework."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize processor
    processor = ClipBatchProcessor(
        output_path="embeddings.parquet",
        start_idx=0,
        end_idx=1000,
        batch_size=50,
        checkpoint_size=100,
        model_name="ViT-bigG-14",
        dataset_name="laion/relaion2B-en-research-safe"
    )
    
    # Run processing
    await processor.run()

if __name__ == "__main__":
    asyncio.run(main()) 
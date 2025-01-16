import abc
import asyncio
from asyncio import Queue, Semaphore
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Generic, List, Optional, Tuple, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm

# Generic type variables for input and output data
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

class BatchProcessor(Generic[InputType, OutputType], abc.ABC):
    """Abstract base class for batch processing with async queues and checkpointing.
    
    This class provides a framework for:
    1. Loading data from an iterator (e.g. HuggingFace datasets)
    2. Processing data in batches with user-defined logic
    3. Automatic checkpointing to parquet files
    """

    def __init__(
        self,
        output_path: str,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        batch_size: int = 50,
        checkpoint_size: int = 1000,
        max_concurrent_tasks: int = 50,
    ):
        """Initialize the batch processor.
        
        Args:
            output_path: Path to save the output parquet file
            start_idx: Starting index in the dataset
            end_idx: Ending index in the dataset (if None, process all data)
            batch_size: Number of items to process in each batch
            checkpoint_size: Number of results to accumulate before checkpointing
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.output_path = Path(output_path)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size
        self.checkpoint_size = checkpoint_size
        self.semaphore = Semaphore(max_concurrent_tasks)
        
        # Initialize queues
        self.load_queue: Queue = Queue(maxsize=500)
        self.process_queue: Queue = Queue(maxsize=500)
        self.write_queue: Queue = Queue(maxsize=500)
        
        # Initialize state
        self._setup_output_file()

    def _setup_output_file(self):
        """Set up the output parquet file if it doesn't exist."""
        if not self.output_path.exists():
            # Create empty parquet file with schema
            empty_df = pd.DataFrame({
                'idx': pd.Series([], dtype='int64'),
                'input': pd.Series([], dtype='object'),
                'output': pd.Series([], dtype='object')
            })
            table = pa.Table.from_pandas(empty_df)
            pq.write_table(table, self.output_path)

    @abc.abstractmethod
    async def do_data_loading(self) -> AsyncIterator[Tuple[int, InputType]]:
        """Load data from source and yield (index, input) tuples."""
        raise NotImplementedError

    @abc.abstractmethod
    async def do_batch_processing(
        self, 
        batch: List[Tuple[int, InputType]]
    ) -> List[Tuple[int, OutputType]]:
        """Process a batch of inputs and return corresponding outputs."""
        raise NotImplementedError

    async def _load_worker(self):
        """Worker to load data and put it in the process queue."""
        try:
            async for idx, input_data in self.do_data_loading():
                if self.end_idx and idx >= self.end_idx:
                    break
                if idx >= self.start_idx:
                    await self.load_queue.put((idx, input_data))
        except asyncio.CancelledError:
            return
        finally:
            # Signal end of data
            await self.load_queue.put((None, None))

    async def _process_worker(self):
        """Worker to process batches from the process queue."""
        try:
            batch = []
            while True:
                try:
                    idx, input_data = await self.load_queue.get()
                    if idx is None:  # End of data
                        break
                        
                    batch.append((idx, input_data))
                    
                    if len(batch) >= self.batch_size:
                        results = await self.do_batch_processing(batch)
                        for result in results:
                            await self.write_queue.put(result)
                        batch = []
                finally:
                    self.load_queue.task_done()
                    
            # Process remaining items in batch
            if batch:
                results = await self.do_batch_processing(batch)
                for result in results:
                    await self.write_queue.put(result)
                    
            # Signal end of processing
            await self.write_queue.put((None, None))
        except asyncio.CancelledError:
            return

    async def _write_worker(self):
        """Worker to write results to parquet file."""
        try:
            buffer = []
            while True:
                idx, output = await self.write_queue.get()
                if idx is None:  # End of data
                    break
                    
                buffer.append({
                    'idx': idx,
                    'output': output
                })
                
                if len(buffer) >= self.checkpoint_size:
                    await self._checkpoint_results(buffer)
                    buffer = []
                    
            # Write remaining results
            if buffer:
                await self._checkpoint_results(buffer)
        except asyncio.CancelledError:
            if buffer:
                await self._checkpoint_results(buffer)
        finally:
            self.write_queue.task_done()

    async def _checkpoint_results(self, results: List[Dict[str, Any]]):
        """Checkpoint results to parquet file."""
        df = pd.DataFrame(results)
        table = pa.Table.from_pandas(df)
        
        # Append to existing parquet file
        pq.write_table(table, self.output_path, append=True)

    async def run(self):
        """Run the batch processing pipeline."""
        # Start workers
        load_task = asyncio.create_task(self._load_worker())
        process_task = asyncio.create_task(self._process_worker())
        write_task = asyncio.create_task(self._write_worker())
        
        # Wait for all tasks to complete
        await asyncio.gather(load_task, process_task, write_task) 
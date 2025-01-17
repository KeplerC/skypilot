from typing import AsyncIterator, Generic, List, Optional, Tuple, TypeVar
import asyncio
import abc

from .base import BatchProcessor, InputType

ModelInputType = TypeVar('ModelInputType')
ModelOutputType = TypeVar('ModelOutputType')

class BatchInferenceProcessor(BatchProcessor[InputType, ModelOutputType], abc.ABC):
    """Abstract base class for model inference tasks.
    
    This class adds model-specific functionality to BatchProcessor:
    1. Model loading and setup
    2. Data preprocessing
    3. Batch inference with the model
    4. Common patterns for handling ML model inputs/outputs
    
    To use this class, implement:
    - setup_model: Initialize your ML model
    - preprocess_input: Convert raw input to model input format
    - run_model_inference: Run the model on preprocessed inputs
    - get_dataset_iterator: Get an iterator over your dataset
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize the batch inference processor.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run inference on ('cuda', 'cpu', etc)
            **kwargs: Additional arguments passed to BatchProcessor
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device or ('cuda' if self.is_cuda_available() else 'cpu')
        self.model = None
        
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    @abc.abstractmethod
    async def setup_model(self):
        """Set up the model for inference.
        
        This method should:
        1. Load the model weights
        2. Move the model to the correct device
        3. Set the model to evaluation mode
        4. Initialize any preprocessing/postprocessing components
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    async def preprocess_input(
        self,
        raw_input: InputType
    ) -> Optional[ModelInputType]:
        """Preprocess a single input item for the model.
        
        Args:
            raw_input: Raw input from the dataset
            
        Returns:
            Preprocessed input ready for the model, or None if preprocessing failed
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    async def run_model_inference(
        self,
        model_inputs: List[ModelInputType]
    ) -> List[ModelOutputType]:
        """Run model inference on a batch of preprocessed inputs.
        
        Args:
            model_inputs: List of preprocessed inputs
            
        Returns:
            List of model outputs
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    async def get_dataset_iterator(self) -> AsyncIterator[Tuple[int, InputType]]:
        """Get an iterator over the dataset.
        
        Returns:
            AsyncIterator yielding (index, input) tuples
        """
        raise NotImplementedError
        
    async def do_data_loading(self) -> AsyncIterator[Tuple[int, InputType]]:
        """Implement BatchProcessor's data loading using dataset iterator."""
        async for item in self.get_dataset_iterator():
            yield item
            
    async def do_batch_processing(
        self,
        batch: List[Tuple[int, InputType]]
    ) -> List[Tuple[int, ModelOutputType]]:
        """Implement BatchProcessor's batch processing using the model pipeline."""
        if self.model is None:
            await self.setup_model()
            
        # Preprocess inputs in parallel
        preprocessed = await asyncio.gather(*[
            self.preprocess_input(input_data)
            for _, input_data in batch
        ])
        
        # Filter out failed preprocessings
        valid_items = [
            (idx, prep_input)
            for (idx, _), prep_input in zip(batch, preprocessed)
            if prep_input is not None
        ]
        
        if not valid_items:
            return []
            
        # Run model inference
        indices, model_inputs = zip(*valid_items)
        outputs = await self.run_model_inference(list(model_inputs))
        return list(zip(indices, outputs)) 
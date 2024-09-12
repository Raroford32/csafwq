import os
import logging
from vllm import LLM

logger = logging.getLogger(__name__)

def load_model(config):
    logger.info(f"Loading model from directory: {config.model_name}")
    model_dir = config.model_name
    
    try:
        model = LLM(
            model=model_dir,
            tensor_parallel_size=config.tensor_parallel_size,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization,
            load_format='auto'  # Added this line to auto-detect the model format
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")
        raise

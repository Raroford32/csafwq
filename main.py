import sys
import logging
import os
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.path}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Contents of current directory: {os.listdir('.')}")

# Check Ray installation
ray_spec = importlib.util.find_spec("ray")
if ray_spec is not None:
    logger.info(f"Ray is installed. Location: {ray_spec.origin}")
    import ray
    logger.info(f"Ray version: {ray.__version__}")
    logger.info("Initializing Ray...")
    try:
        ray.init()
        logger.info("Ray initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {str(e)}")
else:
    logger.error("Ray is not installed or not found in the Python path")

# Check vLLM installation
vllm_spec = importlib.util.find_spec("vllm")
if vllm_spec is not None:
    logger.info(f"vLLM is installed. Location: {vllm_spec.origin}")
    import vllm
    logger.info(f"vLLM version: {vllm.__version__}")
else:
    logger.error("vLLM is not installed or not found in the Python path")

import asyncio
from distributed_inference import DistributedInferenceEngine
from api import start_api_server
from config import Config
from model_loader import load_model
from memory_manager import MemoryManager

async def main():
    try:
        config = Config()
        memory_manager = MemoryManager(config.total_ram)
        
        logger.info("Loading model...")
        model = load_model(config)
        
        logger.info("Initializing distributed inference engine...")
        inference_engine = DistributedInferenceEngine(model, config)
        
        logger.info("Starting API server...")
        await start_api_server(inference_engine)
    except Exception as e:
        logger.exception(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

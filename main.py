import sys
import logging
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.path}")

try:
    logger.info(f"Ray version: {ray.__version__}")
    logger.info("Initializing Ray...")
    ray.init()
    logger.info("Ray initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Ray: {str(e)}")
    sys.exit(1)

try:
    import vllm
    logger.info(f"vLLM successfully imported. Version: {vllm.__version__}")
except ImportError as e:
    logger.error(f"Failed to import vLLM: {str(e)}")
    sys.exit(1)

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

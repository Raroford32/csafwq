import sys
import logging
import os
import importlib.util
import asyncio
import time
from distributed_inference import DistributedInferenceEngine
from api import start_api_server
from config import Config
from model_loader import load_model
from memory_manager import MemoryManager

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
    
    # Print Ray configuration before initialization
    logger.info("Ray configuration:")
    for attr in dir(ray):
        if not attr.startswith("_"):
            try:
                value = getattr(ray, attr)
                if not callable(value):
                    logger.info(f"  {attr}: {value}")
            except Exception as e:
                logger.warning(f"  Unable to get value for {attr}: {str(e)}")
    
    logger.info("Available system resources:")
    import psutil
    logger.info(f"  CPU cores: {psutil.cpu_count()}")
    logger.info(f"  Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        logger.info(f"  Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}, Memory: {gpu.memoryTotal} MB")
    except ImportError:
        logger.warning("GPUtil not installed, skipping GPU information")
    
    logger.info("Initializing Ray...")
    try:
        ray.init(ignore_reinit_error=True, logging_level=logging.DEBUG, log_to_driver=True)
        logger.info("Ray initialized successfully")
        logger.info(f"Ray resources after initialization: {ray.available_resources()}")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {str(e)}")
        logger.error("Ray configuration after failed initialization:")
        for attr in dir(ray):
            if not attr.startswith("_"):
                try:
                    value = getattr(ray, attr)
                    if not callable(value):
                        logger.error(f"  {attr}: {value}")
                except Exception as e:
                    logger.warning(f"  Unable to get value for {attr}: {str(e)}")
        
        # Additional error information
        logger.error("Ray initialization error details:")
        logger.error(f"Ray version: {ray.__version__}")
        logger.error(f"Python version: {sys.version}")
        logger.error(f"Operating system: {os.name}")
        logger.error(f"Platform: {sys.platform}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        
        # Check for Ray-specific environment variables
        ray_env_vars = [var for var in os.environ if var.startswith('RAY_')]
        if ray_env_vars:
            logger.error("Ray-specific environment variables:")
            for var in ray_env_vars:
                logger.error(f"  {var}: {os.environ[var]}")
        else:
            logger.error("No Ray-specific environment variables found")
        
        sys.exit(1)  # Exit if Ray initialization fails
else:
    logger.error("Ray is not installed or not found in the Python path")
    sys.exit(1)  # Exit if Ray is not installed

# Check vLLM installation
vllm_spec = importlib.util.find_spec("vllm")
if vllm_spec is not None:
    logger.info(f"vLLM is installed. Location: {vllm_spec.origin}")
    import vllm
    logger.info(f"vLLM version: {vllm.__version__}")
else:
    logger.error("vLLM is not installed or not found in the Python path")
    sys.exit(1)  # Exit if vLLM is not installed

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
        sys.exit(1)

if __name__ == "__main__":
    start_time = time.time()
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
    finally:
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

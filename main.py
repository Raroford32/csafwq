import logging
import asyncio
import time
from config import Config
from model_loader import load_model
from distributed_inference import DistributedInferenceEngine
from memory_manager import MemoryManager

print("Script execution started")

logging.basicConfig(level=logging.DEBUG, filename='main_debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Logging configured")
logger.info("Logging configured")

async def main():
    try:
        print("Starting main function")
        logger.info("Starting main function")
        
        print("Creating Config instance...")
        logger.info("Creating Config instance...")
        config = Config()
        print("Config instance created successfully")
        logger.info("Config instance created successfully")

        print("Initializing MemoryManager...")
        logger.info("Initializing MemoryManager...")
        memory_manager = MemoryManager(config.total_ram_gb)
        print("MemoryManager initialized successfully")
        logger.info("MemoryManager initialized successfully")

        print("Loading model...")
        logger.info("Loading model...")
        model = load_model(config)
        print("Model loaded successfully")
        logger.info("Model loaded successfully")

        print("Initializing DistributedInferenceEngine...")
        logger.info("Initializing DistributedInferenceEngine...")
        inference_engine = DistributedInferenceEngine(model, config)
        print("DistributedInferenceEngine initialized successfully")
        logger.info("DistributedInferenceEngine initialized successfully")

        # Test the inference engine
        test_prompt = "Explain the concept of distributed computing in one sentence:"
        print(f"Testing inference engine with prompt: {test_prompt}")
        logger.info(f"Testing inference engine with prompt: {test_prompt}")
        response = await inference_engine.generate(test_prompt)
        print(f"Generated response: {response}")
        logger.info(f"Generated response: {response}")

        print("Initialization complete")
        logger.info("Initialization complete")

    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        logger.exception(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        print("Starting asyncio.run(main())")
        asyncio.run(main())
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        logger.exception(f"An unexpected error occurred: {str(e)}")
    finally:
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

import asyncio
from distributed_inference import DistributedInferenceEngine
from api import start_api_server
from config import Config
from model_loader import load_model
from memory_manager import MemoryManager

async def main():
    config = Config()
    memory_manager = MemoryManager(config.total_ram)
    
    print("Loading model...")
    model = load_model(config)
    
    print("Initializing distributed inference engine...")
    inference_engine = DistributedInferenceEngine(model, config)
    
    print("Starting API server...")
    await start_api_server(inference_engine)

if __name__ == "__main__":
    asyncio.run(main())

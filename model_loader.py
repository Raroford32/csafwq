import os
from vllm import LLM

def load_model(config):
    print(f"Loading model from directory: {config.model_name}")
    model_dir = config.model_name
    safetensor_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensor')]
    if not safetensor_files:
        raise ValueError(f"No .safetensor files found in {model_dir}")
    
    print(f"Found {len(safetensor_files)} .safetensor files")
    
    model = LLM(
        model=model_dir,
        tensor_parallel_size=config.tensor_parallel_size,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    print("Model loaded successfully")
    return model

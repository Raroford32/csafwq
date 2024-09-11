from vllm import LLM

def load_model(config):
    print(f"Loading model: {config.model_name}")
    model = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    print("Model loaded successfully")
    return model

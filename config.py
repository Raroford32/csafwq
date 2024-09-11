class Config:
    def __init__(self):
        self.model_name = "meta-llama/Llama-2-70b-chat-hf"  # Example model, replace with your chosen model
        self.num_gpus = 8
        self.num_cpu_cores = 200
        self.total_ram = 1700  # in GB
        self.tensor_parallel_size = 8  # Distribute model across all 8 GPUs
        self.max_num_batched_tokens = 4096  # Adjust based on your requirements
        self.max_num_seqs = 256  # Adjust based on your requirements
        self.gpu_memory_utilization = 0.95  # Adjust based on your requirements
        self.api_host = "0.0.0.0"
        self.api_port = 8000

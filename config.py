class Config:
    def __init__(self):
        self.model_name = "facebook/opt-350m"  # Changed to a smaller model available on Hugging Face
        self.num_gpus = 8
        self.num_cpu_cores = 200
        self.total_ram = 1700  # in GB
        self.tensor_parallel_size = 2  # Reduced for smaller model
        self.max_num_batched_tokens = 4096  # Adjusted for smaller model
        self.max_num_seqs = 128  # Adjusted for smaller model
        self.gpu_memory_utilization = 0.90  # Increased slightly for smaller model
        self.api_host = "0.0.0.0"
        self.api_port = 8000

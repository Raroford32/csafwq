import os

class Config:
    def __init__(self):
        self.model_name = "mlabonne/Hermes-3-Llama-3.1-70B-lorablated"  # Updated to use HuggingFace model identifier
        self.tensor_parallel_size = 8  # Utilizing all 8 GPUs
        self.max_num_batched_tokens = 32768  # Increased to accommodate larger context
        self.max_num_seqs = 128  # Reduced to balance with the increased token count
        self.gpu_memory_utilization = 0.95
        self.num_cpu_cores = 200  # Utilizing all 200 CPU cores
        self.total_ram_gb = 1700  # Total available RAM

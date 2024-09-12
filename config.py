class Config:

    def __init__(self):
        self.model_name = "/home/ubuntu/gwssgv11/csafwq/mlabonne_Hermes-3-Llama-3.1-70B-lorablated"
        self.num_gpus = 8
        self.num_cpu_cores = 200
        self.total_ram = 1700  # in GB
        self.tensor_parallel_size = 8  # Utilize all 8 GPUs
        self.max_num_batched_tokens = 8192  # Increased for larger model
        self.max_num_seqs = 64  # Reduced to accommodate larger model
        self.gpu_memory_utilization = 0.85  # Slightly reduced to leave some headroom
        self.api_host = "0.0.0.0"
        self.api_port = 8000

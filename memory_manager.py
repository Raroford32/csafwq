import psutil

class MemoryManager:
    def __init__(self, total_ram_gb):
        self.total_ram_gb = total_ram_gb
        self.total_ram_bytes = total_ram_gb * 1024 * 1024 * 1024

    def get_available_memory(self):
        return psutil.virtual_memory().available

    def get_used_memory(self):
        return psutil.virtual_memory().used

    def get_memory_usage_percentage(self):
        return psutil.virtual_memory().percent

    def check_memory_threshold(self, threshold_percentage=90):
        if self.get_memory_usage_percentage() > threshold_percentage:
            print(f"Warning: Memory usage is above {threshold_percentage}%")
            return True
        return False

    def log_memory_stats(self):
        print(f"Total RAM: {self.total_ram_gb} GB")
        print(f"Available memory: {self.get_available_memory() / (1024 * 1024 * 1024):.2f} GB")
        print(f"Used memory: {self.get_used_memory() / (1024 * 1024 * 1024):.2f} GB")
        print(f"Memory usage: {self.get_memory_usage_percentage()}%")

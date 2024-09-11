import time
import psutil
import GPUtil

def log_system_stats():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")

    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / (1024 * 1024 * 1024):.2f} GB")

    # GPU usage
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        print(f"GPU {i} Usage: {gpu.load * 100}%")
        print(f"GPU {i} Memory Usage: {gpu.memoryUsed / gpu.memoryTotal * 100:.2f}%")

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

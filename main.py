import subprocess
import sys
import shutil

def main():
    try:
        model_name = "mlabonne/Hermes-3-Llama-3.1-70B-lorablated"
        vllm_path = shutil.which('vllm')
        if not vllm_path:
            raise FileNotFoundError("vLLM executable not found in PATH")
        command = f"{vllm_path} serve {model_name} --host 0.0.0.0 --port 8000"
        print(f"Starting vLLM server with command: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running vLLM serve: {e}", file=sys.stderr)
        print(f"Command output: {e.output}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

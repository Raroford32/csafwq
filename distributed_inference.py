from vllm import SamplingParams
from concurrent.futures import ThreadPoolExecutor

class DistributedInferenceEngine:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.num_cpu_cores)

    async def generate(self, prompt, max_tokens=100, temperature=0.7, top_p=0.95):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        outputs = await self.model.generate_async([prompt], sampling_params)
        return outputs[0].outputs[0].text

    async def batch_generate(self, prompts, max_tokens=100, temperature=0.7, top_p=0.95):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        outputs = await self.model.generate_async(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def parallel_generate(self, prompts, max_tokens=100, temperature=0.7, top_p=0.95):
        futures = [self.executor.submit(self.generate, prompt, max_tokens, temperature, top_p) for prompt in prompts]
        return [future.result() for future in futures]

from vllm import SamplingParams
import asyncio

class DistributedInferenceEngine:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    async def generate(self, prompt, max_tokens=100, temperature=0.7, top_p=0.95):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = await self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    async def batch_generate(self, prompts, max_tokens=100, temperature=0.7, top_p=0.95):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = await self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

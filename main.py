import logging
import asyncio
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config

logging.basicConfig(level=logging.DEBUG, filename='main_debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPUInferenceEngine:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    async def generate(self, prompt, max_tokens=100, temperature=0.7, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    async def batch_generate(self, prompts, max_tokens=100, temperature=0.7, top_p=0.95):
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, max_tokens, temperature, top_p)
            results.append(result)
        return results

async def main():
    try:
        logger.info("Starting main function")
        logger.info("Creating Config instance...")
        config = Config()
        logger.info("Config instance created successfully")

        logger.info("Initializing CPU inference engine...")
        inference_engine = CPUInferenceEngine(config.model_name)
        logger.info("CPU inference engine initialized successfully")

        # Test the inference engine
        test_prompts = [
            "Explain the concept of distributed computing in one sentence:",
            "What are the benefits of using multiple CPUs for inference?",
            "How does Hugging Face Transformers help with large language model inference?"
        ]
        logger.info(f"Generating responses for {len(test_prompts)} test prompts")
        responses = await inference_engine.batch_generate(test_prompts)
        for prompt, response in zip(test_prompts, responses):
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")

    except Exception as e:
        logger.exception(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
    finally:
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

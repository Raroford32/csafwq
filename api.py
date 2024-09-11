from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95

inference_engine = None

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        result = await inference_engine.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_generate")
async def batch_generate(request: BatchGenerateRequest):
    try:
        results = await inference_engine.batch_generate(
            request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"generated_texts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def start_api_server(inference_eng):
    global inference_engine
    inference_engine = inference_eng
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()

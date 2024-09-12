import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    vllm_url = "http://localhost:8000/v1/completions"
    
    # Convert chat messages to a prompt
    prompt = " ".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    vllm_request = {
        "prompt": prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "n": request.n,
        "max_tokens": request.max_tokens,
        "stop": request.stop,
        "stream": request.stream
    }
    
    try:
        response = requests.post(vllm_url, json=vllm_request)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with vLLM server: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

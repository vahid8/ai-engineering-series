"""Episode 3 — your own LLM Gateway.

A tiny HTTP service that wraps LiteLLM. Every app you build calls THIS instead
of embedding provider keys and SDKs everywhere. Keys live in one place; swap
models or providers without touching a single app.

The trick: LiteLLM already returns an OpenAI-shaped response, so our gateway
just forwards it — which means it's OpenAI-compatible. You can point the same
OpenAI SDK from Episode 1 straight at this gateway (see 03_client.py).

Run it:
    uv run --env-file .env python episodes/03_gateway.py
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from litellm import acompletion
from pydantic import BaseModel

logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # quiet startup warnings


def load_key(env_name: str) -> None:
    """Keys live in ONE place — the gateway's environment."""
    if os.environ.get(env_name):
        return
    if key_file := os.environ.get(f"{env_name}_FILE"):
        os.environ[env_name] = Path(key_file).expanduser().read_text().strip()


load_key("GEMINI_API_KEY")
load_key("GROQ_API_KEY")

app = FastAPI(title="My LLM Gateway")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible: any OpenAI client can call this. We hand the model +
    messages to LiteLLM and return its (already OpenAI-shaped) response."""
    response = await acompletion(model=req.model, messages=req.messages)
    return response.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

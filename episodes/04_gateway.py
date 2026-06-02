"""Episode 4 — streaming responses from your gateway (tokens live, like ChatGPT).

This is the Ep3 gateway plus one new ability: streaming. When a request asks for
stream=True, instead of waiting for the whole answer we forward LiteLLM's token
chunks the moment they arrive, in the exact OpenAI SSE ("data: ...") format — so
the same OpenAI SDK from Episode 1 can stream from our gateway too.

Run it:
    uv run --env-file .env python episodes/04_gateway.py
"""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
    stream: bool = False  # NEW: clients can now ask for a live stream


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible. stream=False behaves exactly like Episode 3."""
    if not req.stream:
        response = await acompletion(model=req.model, messages=req.messages)
        return response.model_dump()

    # Streaming: forward each token chunk as an OpenAI-style SSE event.
    async def event_stream():
        chunks = await acompletion(model=req.model, messages=req.messages, stream=True)
        async for chunk in chunks:
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

"""Episode 5 — tokens & cost: see what every call really costs.

Builds on the Ep4 streaming gateway. Now every call is measured: how many tokens
it used (prompt vs completion), what that cost in dollars, and how long it took.
We keep a running tally per model and expose it at GET /usage — so you can see
exactly why your LLM bill grows (hint: it's mostly the input tokens you forgot about).

Run it:
    uv run --env-file .env python episodes/05_gateway.py
"""

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from litellm import acompletion, cost_per_token
from pydantic import BaseModel

logging.getLogger("LiteLLM").setLevel(logging.ERROR)


def load_key(env_name: str) -> None:
    if os.environ.get(env_name):
        return
    if key_file := os.environ.get(f"{env_name}_FILE"):
        os.environ[env_name] = Path(key_file).expanduser().read_text().strip()


load_key("GEMINI_API_KEY")
load_key("GROQ_API_KEY")

app = FastAPI(title="My LLM Gateway")

# Running bill, per model.
USAGE = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0,
                             "completion_tokens": 0, "cost_usd": 0.0})


def track(model: str, prompt_toks: int, completion_toks: int, latency_s: float) -> None:
    """Record tokens, cost, and latency for one call — and print it."""
    try:
        # LiteLLM knows each model's per-token price. Input and output differ —
        # output tokens are usually several times more expensive.
        in_cost, out_cost = cost_per_token(
            model=model, prompt_tokens=prompt_toks, completion_tokens=completion_toks
        )
    except Exception:
        in_cost = out_cost = 0.0
    row = USAGE[model]
    row["requests"] += 1
    row["prompt_tokens"] += prompt_toks
    row["completion_tokens"] += completion_toks
    row["cost_usd"] += in_cost + out_cost
    print(f"{model}: {prompt_toks} in (${in_cost:.6f}) + {completion_toks} out "
          f"(${out_cost:.6f}) = ${in_cost + out_cost:.6f}  ({latency_s * 1000:.0f} ms)")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    max_tokens: int | None = None  # cap the OUTPUT length (and so the output cost)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/usage")
def usage():
    """The running bill — total tokens and dollars per model."""
    return USAGE


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.stream:
        start = time.perf_counter()
        response = await acompletion(model=req.model, messages=req.messages,
                                     max_tokens=req.max_tokens)
        u = response.usage
        track(req.model, u.prompt_tokens, u.completion_tokens, time.perf_counter() - start)
        return response.model_dump()

    async def event_stream():
        start = time.perf_counter()
        # include_usage tells the provider to send a final chunk carrying token counts.
        chunks = await acompletion(model=req.model, messages=req.messages,
                                   max_tokens=req.max_tokens,
                                   stream=True, stream_options={"include_usage": True})
        usage_seen = None
        async for chunk in chunks:
            if getattr(chunk, "usage", None):
                usage_seen = chunk.usage
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
        if usage_seen:
            track(req.model, usage_seen.prompt_tokens, usage_seen.completion_tokens,
                  time.perf_counter() - start)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

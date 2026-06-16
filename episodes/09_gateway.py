"""Episode 9 — embeddings: turning text into vectors.

Builds on the Ep8 multi-user gateway. Until now the gateway has done one thing:
chat completions. This episode adds a SECOND capability — embeddings — and it's
the foundation of search, recommendations, and RAG.

An embedding turns a piece of text into a list of numbers (a vector) that
captures its MEANING. Texts that mean similar things land close together; texts
that mean different things land far apart. Measure "close" with cosine
similarity and you have semantic search — matching on meaning, not keywords.

The gateway barely changes: one new endpoint, `POST /v1/embeddings`, behind the
exact same gate (API key -> rate limit -> budget) as chat. LiteLLM gives us
`aembedding`, the embedding twin of `acompletion`.

Run it (in another terminal):
    uv run --env-file .env python episodes/09_gateway.py
"""

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from litellm import acompletion, aembedding, cost_per_token
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

# --- API KEYS: OUR gateway's keys -> the user each one belongs to (from Ep8). ---
API_KEYS = {
    "sk-demo-alice": "alice",  # a normal user (generous budget)
    "sk-demo-carol": "carol",
    "sk-demo-bob": "bob",
}

# --- Rate limit + budget gates (Ep8). Embeddings go through the SAME gate. ------
RATE_LIMIT = 5
RATE_WINDOW = 30  # seconds
CALLS: dict[str, list[float]] = defaultdict(list)  # user -> recent call timestamps
BUDGET_USD = {"alice": 1.00, "carol": 1.00, "bob": 0.0018}

# The running bill per user (Ep5 cost tracker).
USAGE = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0,
                             "completion_tokens": 0, "cost_usd": 0.0})


def authenticate(authorization: str = Header(None)) -> str:
    """Turn the `Authorization: Bearer <key>` header into a user — or 401."""
    key = (authorization or "").removeprefix("Bearer ").strip()
    user = API_KEYS.get(key)
    if not user:
        raise HTTPException(status_code=401, detail="invalid API key")
    return user


def enforce_limits(user: str) -> None:
    """The two gates, run BEFORE we spend anything on the provider."""
    now = time.time()
    recent = CALLS[user] = [t for t in CALLS[user] if now - t < RATE_WINDOW]
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(status_code=429,
                            detail=f"rate limit: {RATE_LIMIT} requests / {RATE_WINDOW}s")
    recent.append(now)
    if USAGE[user]["cost_usd"] >= BUDGET_USD.get(user, 0.0):
        raise HTTPException(status_code=402,
                            detail=f"budget exceeded: ${BUDGET_USD.get(user, 0.0):.4f}")


def track(user: str, model: str, prompt_toks: int, completion_toks: int,
          latency_s: float) -> None:
    """Add this call to the user's running bill — and print it."""
    try:
        in_cost, out_cost = cost_per_token(
            model=model, prompt_tokens=prompt_toks, completion_tokens=completion_toks)
    except Exception:
        in_cost = out_cost = 0.0
    row = USAGE[user]
    row["requests"] += 1
    row["prompt_tokens"] += prompt_toks
    row["completion_tokens"] += completion_toks
    row["cost_usd"] += in_cost + out_cost
    print(f"[{user}] {model}: {prompt_toks}+{completion_toks} tok "
          f"= ${in_cost + out_cost:.6f}  (total ${row['cost_usd']:.6f}, "
          f"{latency_s * 1000:.0f} ms)")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    max_tokens: int | None = None
    response_format: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


# NEW: an embeddings request is tiny — a model and the text(s) to embed.
class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    dimensions: int | None = None  # some models let you ask for a shorter vector


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/usage")
def usage():
    """The owner's view: the running bill for every user. (Protect this for real.)"""
    return USAGE


# NEW: the embeddings endpoint — same gate as chat, just aembedding instead of
# acompletion. Returns the OpenAI-shaped payload so any OpenAI SDK can read it.
@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest, user: str = Depends(authenticate)):
    enforce_limits(user)  # SAME 401 / 429 / 402 gate as chat — one door for all.
    start = time.perf_counter()
    response = await aembedding(model=req.model, input=req.input,
                                dimensions=req.dimensions)
    # Embeddings bill on input tokens only — there's no completion to generate.
    track(user, req.model, response.usage.prompt_tokens, 0,
          time.perf_counter() - start)
    return response.model_dump()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, user: str = Depends(authenticate)):
    enforce_limits(user)  # 429 / 402 BEFORE we call the provider and spend money.

    if not req.stream:
        start = time.perf_counter()
        response = await acompletion(model=req.model, messages=req.messages,
                                     max_tokens=req.max_tokens,
                                     response_format=req.response_format,
                                     tools=req.tools, tool_choice=req.tool_choice)
        u = response.usage
        track(user, req.model, u.prompt_tokens, u.completion_tokens,
              time.perf_counter() - start)
        return response.model_dump()

    async def event_stream():
        start = time.perf_counter()
        chunks = await acompletion(model=req.model, messages=req.messages,
                                   max_tokens=req.max_tokens,
                                   response_format=req.response_format,
                                   tools=req.tools, tool_choice=req.tool_choice,
                                   stream=True, stream_options={"include_usage": True})
        usage_seen = None
        async for chunk in chunks:
            if getattr(chunk, "usage", None):
                usage_seen = chunk.usage
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
        if usage_seen:
            track(user, req.model, usage_seen.prompt_tokens,
                  usage_seen.completion_tokens, time.perf_counter() - start)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

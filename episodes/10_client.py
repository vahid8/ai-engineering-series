"""Episode 10 — RAG part 1: retrieval (chunk -> embed -> store -> retrieve).

RAG = Retrieval-Augmented Generation. You can't paste a whole handbook into a
prompt, so instead you RETRIEVE the few passages that matter and (next episode)
hand those to the model. This episode builds the "retrieval" half — no LLM yet.

The pipeline, all client-side on top of the gateway's /v1/embeddings:
  1) CHUNK   — split each document into small passages.
  2) EMBED   — turn every chunk into a vector (one batched gateway call).
  3) STORE   — keep (doc, chunk, vector) in a list. That list IS our vector store.
  4) RETRIEVE— embed the question, score every chunk by cosine, take the top-k.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/10_gateway.py

Then run this:
    uv run python episodes/10_client.py
"""

import math
import re
import sys

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GATEWAY = "http://127.0.0.1:8000/v1"
MODEL = "gemini/gemini-embedding-001"
DIMS = 768

client = OpenAI(api_key="sk-demo-alice", base_url=GATEWAY)

# Our tiny "knowledge base" — three short docs on different topics. In real life
# these would be files, web pages, or a wiki; the pipeline is identical.
DOCS = {
    "billing.md": (
        "Usage is billed per token, counting both your input and the model's "
        "output. Invoices go out on the first of each month. You can set a hard "
        "spending cap per API key from the dashboard to avoid surprise bills."
    ),
    "limits.md": (
        "Every API key is rate limited to sixty requests per minute. Go over and "
        "the gateway returns a 429 error until the window resets. The limit is per "
        "key, so one busy key never slows down anyone else's."
    ),
    "embeddings.md": (
        "The embeddings endpoint turns a piece of text into a 768-number vector "
        "that captures its meaning. Compare two vectors with cosine similarity to "
        "see how related they are. It powers search, clustering, and RAG."
    ),
}


def embed(texts: list[str]) -> list[list[float]]:
    """One batched call to the gateway -> one vector per text."""
    resp = client.embeddings.create(model=MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def chunk(text: str, size: int = 160) -> list[str]:
    """Split text into passages of ~`size` characters, never mid-sentence.
    Real chunkers add overlap and respect headings — this is the honest core."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > size:
            chunks.append(current.strip())
            current = ""
        current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


# 1-3) INGEST — chunk every doc, embed every chunk, keep it all in one list.
store = [{"doc": name, "text": c} for name, text in DOCS.items() for c in chunk(text)]
for item, vector in zip(store, embed([item["text"] for item in store])):
    item["vec"] = vector
print(f"ingested {len(DOCS)} docs -> {len(store)} chunks, each a {DIMS}-d vector\n")


# 4) RETRIEVE — embed the question, score every chunk, return the best k.
def retrieve(query: str, k: int = 3) -> list[tuple[float, str, str]]:
    qv = embed([query])[0]
    scored = ((cosine(qv, item["vec"]), item["doc"], item["text"]) for item in store)
    return sorted(scored, reverse=True)[:k]


query = "how many requests can I send before I get blocked?"
print(f"query: {query!r}\n")
print("top chunks (cosine · source · text):")
for score, doc, text in retrieve(query):
    print(f"  {score:.3f}  [{doc}]  {text}")

print("\n-> retrieval found the right passage by MEANING — note 'blocked' and "
      "'60 / minute' never share a word.\n   Next episode: hand these chunks to "
      "the LLM so it answers FROM your docs (that's the 'generation' in RAG).")

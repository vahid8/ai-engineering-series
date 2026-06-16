"""Episode 9 — embeddings: turning text into vectors.

Same OpenAI SDK, same gateway URL — we just call a different method:
`client.embeddings.create(...)` instead of `chat.completions.create(...)`.

Two steps:
  1) See what an embedding IS — one sentence becomes a list of 768 numbers.
  2) Use it: semantic search. Embed a query and a few documents, then rank the
     documents by COSINE SIMILARITY. The best match wins on MEANING, even when it
     shares no words with the query. That ranking is the heart of RAG.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/09_gateway.py

Then run this:
    uv run python episodes/09_client.py
"""

import math
import sys

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GATEWAY = "http://127.0.0.1:8000/v1"
MODEL = "gemini/gemini-embedding-001"
DIMS = 768  # ask for a 768-number vector (the model can go bigger; smaller is cheaper)

client = OpenAI(api_key="sk-demo-alice", base_url=GATEWAY)


def embed(texts: list[str]) -> list[list[float]]:
    """One call to the gateway -> one vector per input text."""
    resp = client.embeddings.create(model=MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    """Similarity = how aligned two vectors point. 1.0 = identical meaning, ~0 = unrelated.
    It's just a dot product divided by the two lengths — no library needed."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / norm


# 1) WHAT AN EMBEDDING IS — text in, a list of numbers out.
print("1) one sentence -> one vector:")
[vector] = embed(["The quick brown fox jumps over the lazy dog."])
print(f"   length: {len(vector)} numbers")
print(f"   first 5: {[round(x, 4) for x in vector[:5]]}")

# 2) SEMANTIC SEARCH — rank documents by meaning, not keywords.
query = "I'm locked out of my account"
docs = [
    "How do I reset my password?",
    "The cat napped in a sunny window all afternoon.",
    "Our return policy lasts thirty days from purchase.",
]

# Embed the query and every document in a single call, then score each doc.
query_vec, *doc_vecs = embed([query] + docs)
ranked = sorted(zip(docs, doc_vecs), key=lambda d: cosine(query_vec, d[1]), reverse=True)

print(f"\n2) semantic search for: {query!r}")
for doc, vec in ranked:
    print(f"   {cosine(query_vec, vec):.3f}  {doc}")

best = ranked[0][0]
print(f"\n   -> best match: {best!r}")
print("   (note: it shares NO words with the query — matched on meaning, not keywords.)")

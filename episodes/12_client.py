"""Episode 12 — RAG part 3: better retrieval (chunk overlap + reranking with MMR).

Plain top-k by cosine has a sneaky flaw: the top results are often near-DUPLICATES.
Ask "what are all the reasons my requests fail?" and cosine happily returns the
same rate-limit passage three times — so the model never even sees the other
reasons, and the answer comes out narrow. Two cheap upgrades fix the retrieval
you feed the model:

  OVERLAP  — neighbouring chunks share a little text, so a fact that straddles a
             chunk boundary still lives whole inside at least one chunk.
  RERANK   — reorder the wide result set for DIVERSITY with MMR (Maximal Marginal
             Relevance): each pick must be relevant to the query AND different
             from what you've already chosen. Same cost as cosine (pure vector
             math), but the top-k now covers the question instead of repeating
             one passage. (A cross-encoder / LLM reranker is the other flavour —
             that one buys precision; MMR buys diversity.)

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/12_gateway.py

Then run this:
    uv run python episodes/12_client.py
"""

import math
import re
import sys

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GATEWAY = "http://127.0.0.1:8000/v1"
EMBED_MODEL = "gemini/gemini-embedding-001"
CHAT_MODEL = "gemini/gemini-2.5-flash"
DIMS = 768

client = OpenAI(api_key="sk-demo-alice", base_url=GATEWAY)

# A few docs that each describe a DIFFERENT way a request can be stopped — so a
# good answer has to pull from several of them, not repeat one.
DOCS = {
    "billing.md": (
        "Usage is billed per token, counting both your input and the model's "
        "output. Invoices go out on the first of each month. To control costs, "
        "set a hard spending cap per API key from the dashboard."
    ),
    "rate-limits.md": (
        "Every API key is rate limited to sixty requests per minute. Go over and "
        "the gateway returns a 429 error until the window resets. The limit is per "
        "key, so one busy key never slows down anyone else's."
    ),
    "refunds.md": (
        "We do not offer refunds for usage already incurred. If a key was blocked "
        "by a spending cap you set, the calls were never made, so there is nothing "
        "to refund. Annual plans can be cancelled for a prorated credit."
    ),
    "embeddings.md": (
        "The embeddings endpoint turns text into a 768-number vector that captures "
        "its meaning. Compare two vectors with cosine similarity to see how related "
        "they are. It powers search, clustering, and RAG."
    ),
}


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


# UPGRADE 1 — chunk OVERLAP. Carry the tail of each chunk into the next, so a fact
# split across a boundary still appears whole somewhere.
def chunk(text: str, size: int = 120, overlap: int = 40) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > size:
            chunks.append(current.strip())
            current = current[-overlap:]  # the overlap: previous tail carried over
        current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


store = [{"doc": name, "text": c} for name, text in DOCS.items() for c in chunk(text)]
for item, vector in zip(store, embed([item["text"] for item in store])):
    item["vec"] = vector


def retrieve(query_vec: list[float], n: int = 6) -> list[dict]:
    """Wide, cheap recall: the n chunks closest to the query by cosine."""
    return sorted(store, key=lambda it: cosine(query_vec, it["vec"]), reverse=True)[:n]


# UPGRADE 2 — MMR reranking. Greedily pick the chunk that maximises
# (relevance to query) minus (similarity to what we've already picked).
def mmr(query_vec: list[float], pool: list[dict], k: int = 3, lam: float = 0.6) -> list[dict]:
    selected: list[dict] = []
    pool = list(pool)
    while pool and len(selected) < k:
        best, best_score = None, -1e9
        for c in pool:
            relevance = cosine(query_vec, c["vec"])
            redundancy = max((cosine(c["vec"], s["vec"]) for s in selected), default=0.0)
            score = lam * relevance - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        pool.remove(best)
    return selected


SYSTEM = (
    "You answer questions using ONLY the context passages provided. "
    "Each passage is tagged with its source file like [rate-limits.md]. "
    "Cite the source file in square brackets after each fact you use. "
    "If the answer is not in the context, reply exactly: "
    '"I don\'t know based on the provided documents." Do not use outside knowledge.'
)


def generate(query: str, chunks: list[dict]) -> str:
    context = "\n".join(f"[{c['doc']}] {c['text']}" for c in chunks)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content.strip()


print(f"indexed {len(DOCS)} docs -> {len(store)} chunks (with overlap)\n")

query = "what are all the reasons my requests might fail or be stopped?"
query_vec = embed([query])[0]
wide = retrieve(query_vec, n=6)            # one wide, cheap recall pass

plain = wide[:3]                           # naive: just the top-3 by cosine
diverse = mmr(query_vec, wide, k=3)        # reranked for diversity

print(f"Q: {query}\n")
print(f"  plain top-3 (cosine) : {[c['doc'] for c in plain]}")
print(f"  reranked top-3 (MMR) : {[c['doc'] for c in diverse]}\n")

# Same question, same model — the ONLY difference is which chunks we retrieved.
print("answer from PLAIN cosine context:")
print(f"  {generate(query, plain)}\n")
print("answer from MMR-reranked context:")
print(f"  {generate(query, diverse)}\n")

print("-> cosine returned the SAME passage three times, so the answer only knew "
      "about rate limits.\n   MMR reranked for diversity — rate limits + spending "
      "cap + billing — so the answer is COMPLETE.\n   Better retrieval beats a "
      "bigger prompt: feed the model the right, non-redundant passages.")

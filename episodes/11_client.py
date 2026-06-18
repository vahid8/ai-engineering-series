"""Episode 11 — RAG part 2: grounding + citations (the G in RAG).

Last episode we built RETRIEVAL: chunk -> embed -> store -> retrieve top-k.
We pulled back the right passages, but we never called the LLM. This episode
adds GENERATION: hand those passages to the model and have it answer FROM your
docs — grounded in the retrieved text, with a citation for every claim, and an
honest "I don't know" when the answer simply isn't in the documents.

The whole RAG loop, end to end:
  RETRIEVE  -> embed the question, cosine vs every chunk, take the top-k.
  AUGMENT   -> stuff those chunks into the prompt as the ONLY allowed source.
  GENERATE  -> the LLM writes the answer, citing the source file per claim.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/11_gateway.py

Then run this:
    uv run python episodes/11_client.py
"""

import math
import re
import sys

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GATEWAY = "http://127.0.0.1:8000/v1"
EMBED_MODEL = "gemini/gemini-embedding-001"  # to RETRIEVE (Ep9/10)
CHAT_MODEL = "gemini/gemini-2.5-flash"       # to GENERATE (since Ep3)
DIMS = 768

client = OpenAI(api_key="sk-demo-alice", base_url=GATEWAY)

# Same tiny "knowledge base" as last episode — three short docs on different
# topics. In real life these are files, web pages, or a wiki; pipeline identical.
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


# ---- RETRIEVAL: exactly what we built last episode (Ep10) -------------------
def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def chunk(text: str, size: int = 160) -> list[str]:
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


store = [{"doc": name, "text": c} for name, text in DOCS.items() for c in chunk(text)]
for item, vector in zip(store, embed([item["text"] for item in store])):
    item["vec"] = vector


def retrieve(query: str, k: int = 3) -> list[tuple[float, str, str]]:
    qv = embed([query])[0]
    scored = ((cosine(qv, item["vec"]), item["doc"], item["text"]) for item in store)
    return sorted(scored, reverse=True)[:k]


# ---- GENERATION: the NEW half — answer FROM the retrieved passages ----------
# The system prompt is where grounding lives. We tell the model three rules:
# only use the context, cite the source file, and refuse when it doesn't know.
SYSTEM = (
    "You answer questions using ONLY the context passages provided. "
    "Each passage is tagged with its source file like [limits.md]. "
    "Cite the source file in square brackets after each fact you use. "
    "If the answer is not in the context, reply exactly: "
    '"I don\'t know based on the provided documents." Do not use outside knowledge.'
)


def answer(query: str, k: int = 3) -> None:
    # 1) RETRIEVE the passages that matter.
    hits = retrieve(query, k)
    # 2) AUGMENT — build the context block the model is allowed to use.
    context = "\n".join(f"[{doc}] {text}" for _, doc, text in hits)
    # 3) GENERATE — the model answers, grounded in that context, with citations.
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    print(f"Q: {query}")
    print(f"A: {resp.choices[0].message.content.strip()}")
    print(f"   retrieved from: {', '.join(doc for _, doc, _ in hits)}\n")


print(f"indexed {len(DOCS)} docs -> {len(store)} chunks\n")

# Q1 — IN the docs: a grounded, cited answer straight from limits.md.
answer("how many requests can I send before I get blocked?")

# Q2 — NOT in the docs: retrieval still returns its closest chunks, but the
# model must REFUSE instead of inventing an answer. That's grounding.
answer("can I pay my invoice with PayPal?")

print("-> grounded answers cite the source file; off-topic questions get an "
      "honest 'I don't know' instead of a hallucination.\n   That's RAG: "
      "retrieve, augment, generate — answers you can trust because you can check them.")

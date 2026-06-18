"""Episode 13 — Evaluating your RAG: retrieval recall@k + an LLM judge.

We spent three episodes making RAG better — chunking, retrieval, grounding,
reranking. But "better" has been a vibe. This episode turns it into a NUMBER, so
you can change a knob (chunk size, MMR on/off, the model) and actually KNOW if it
helped instead of guessing.

We score two things, the two halves of RAG:

  RETRIEVAL — recall@k: for a question whose answer we know lives in doc X, did
              X show up in the retrieved chunks? Average over a small labelled
              eval set -> a recall score.
  ANSWER    — an LLM-as-JUDGE: a second model call reads the question, the answer
              we produced, and a reference, and returns PASS / FAIL with a reason.
              Average -> an answer-correctness score.

Together they're your RAG's report card. Re-run it after any change to catch
regressions before your users do.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/13_gateway.py

Then run this:
    uv run python episodes/13_client.py
"""

import json
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

# The same knowledge base as last episode.
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


# ---- The RAG pipeline from Ep10-12 (chunk+overlap -> embed -> retrieve -> MMR -> ground) ----
def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def chunk(text: str, size: int = 120, overlap: int = 40) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > size:
            chunks.append(current.strip())
            current = current[-overlap:]
        current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


store = [{"doc": name, "text": c} for name, text in DOCS.items() for c in chunk(text)]
for item, vector in zip(store, embed([item["text"] for item in store])):
    item["vec"] = vector


def mmr(query_vec, pool, k=3, lam=0.6):
    selected, pool = [], list(pool)
    while pool and len(selected) < k:
        best, best_score = None, -1e9
        for c in pool:
            rel = cosine(query_vec, c["vec"])
            red = max((cosine(c["vec"], s["vec"]) for s in selected), default=0.0)
            score = lam * rel - (1 - lam) * red
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        pool.remove(best)
    return selected


def retrieve(query: str, k: int = 3) -> list[dict]:
    qv = embed([query])[0]
    wide = sorted(store, key=lambda it: cosine(qv, it["vec"]), reverse=True)[:6]
    return mmr(qv, wide, k=k)


SYSTEM = (
    "You answer questions using ONLY the context passages provided, each tagged "
    "with its source file like [billing.md]. Cite the source file after each fact. "
    "If the answer is not in the context, reply exactly: "
    '"I don\'t know based on the provided documents."'
)


def answer(query: str, chunks: list[dict]) -> str:
    context = "\n".join(f"[{c['doc']}] {c['text']}" for c in chunks)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---- THE EVAL SET — questions with a KNOWN right source + a reference answer ----
# `source=None` means the docs can't answer it: the RIGHT behaviour is to refuse.
EVAL = [
    {"q": "How many requests per minute can one API key make?",
     "source": "rate-limits.md",
     "reference": "Sixty requests per minute per key; going over returns a 429 until the window resets."},
    {"q": "How do I stop my bill from getting too high?",
     "source": "billing.md",
     "reference": "Set a hard spending cap per API key from the dashboard."},
    {"q": "Can I get a refund for tokens I already used?",
     "source": "refunds.md",
     "reference": "No — there are no refunds for usage already incurred; annual plans can be cancelled for a prorated credit."},
    {"q": "Can I pay my invoice with PayPal?",
     "source": None,
     "reference": "The documents say nothing about payment methods, so the system MUST refuse with 'I don't know based on the provided documents.'"},
]


# ---- METRIC 2 — the LLM-as-judge. A second call grades our answer vs a reference. ----
def judge(question: str, ans: str, reference: str) -> dict:
    prompt = (
        "You are grading a RAG system's answer. Be strict.\n"
        f"Question: {question}\n"
        f"Reference (the correct answer): {reference}\n"
        f"Answer to grade: {ans}\n\n"
        "Does the answer match the reference (correct and not missing key facts)?\n"
        'Reply with JSON only: {"verdict": "PASS" or "FAIL", "why": "one short reason"}'
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


print(f"indexed {len(DOCS)} docs -> {len(store)} chunks · running eval on "
      f"{len(EVAL)} questions\n")

retrieval_hits = retrieval_total = answer_pass = 0
for item in EVAL:
    chunks = retrieve(item["q"], k=3)
    got = {c["doc"] for c in chunks}

    # METRIC 1 — recall@k: is the expected source among the retrieved docs?
    recall_mark = "  -  (refusal, n/a)"
    if item["source"] is not None:
        retrieval_total += 1
        hit = item["source"] in got
        retrieval_hits += hit
        recall_mark = f"recall@3 {'HIT ' if hit else 'MISS'} (want {item['source']})"

    # METRIC 2 — generate the answer, then let the judge grade it.
    ans = answer(item["q"], chunks)
    verdict = judge(item["q"], ans, item["reference"])
    answer_pass += verdict["verdict"] == "PASS"

    print(f"Q: {item['q']}")
    print(f"   {recall_mark}")
    print(f"   judge: {verdict['verdict']}  ({verdict['why']})\n")

print("=" * 60)
print(f"RETRIEVAL  recall@3      : {retrieval_hits}/{retrieval_total}")
print(f"ANSWER     judge pass    : {answer_pass}/{len(EVAL)}")
print("=" * 60)
print("\n-> that's your RAG's report card. Now change ONE thing — chunk size, MMR "
      "on/off, the model —\n   re-run, and watch the numbers. Evaluation is how "
      "you improve RAG on purpose instead of by vibes.")

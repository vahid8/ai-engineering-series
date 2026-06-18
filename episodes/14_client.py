"""Episode 14 (FINALE) — point the Episode-1 client at the REAL gateway.

Across thirteen episodes our toy gateway grew one feature at a time — routing,
streaming, cost tracking, JSON, tools, API keys, rate limits, budgets,
embeddings, RAG. Put together, that's a real gateway. And the real,
production-grade version is open source:

    https://github.com/vahid8/llm-gateway

This episode closes the loop. The SAME OpenAI SDK from Episode 1 — only the
base_url changed — now talks to that real gateway running locally, and every
call shows up on its /dashboard with cost, latency, and tokens. No toy gateway
file this time: we run the real one.

--- Run the REAL gateway first (separate repo) ---------------------------------
    git clone https://github.com/vahid8/llm-gateway && cd llm-gateway
    uv sync
    # provider key + an admin key, screen-safe (values never printed):
    export GEMINI_API_KEY="$(cat ~/.secrets/gemini.key)"
    export ADMIN_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
    uv run uvicorn app.main:app            # serves on http://127.0.0.1:8000
    # issue yourself a gateway key (prints sk-... once):
    curl -s -X POST localhost:8000/admin/keys \
      -H "Authorization: Bearer $ADMIN_API_KEY" \
      -H "Content-Type: application/json" -d '{"name":"finale"}'
    export GATEWAY_API_KEY="sk-..."        # paste the minted key

--- Then run this --------------------------------------------------------------
    uv run python episodes/14_client.py
"""

import os

from openai import OpenAI

# The real gateway, running locally. This is the EXACT OpenAI SDK from Episode 1
# — the only thing that changed across the whole series is the base_url. Your
# gateway key comes from the environment; never hard-code it.
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ.get("GATEWAY_API_KEY", "sk-set-your-gateway-key"),
)

# One model string is all you change to switch providers — and every call is
# authenticated, rate-limited, budgeted, cost-tracked, and logged by YOUR gateway.
# (Add "claude-..." or "gpt-4o" here too if you set those provider keys.)
for model in ["gemini-2.5-flash"]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is an LLM gateway?"}],
    )
    print(f"[{model}] {resp.choices[0].message.content}")

print(
    "\n-> Same OpenAI SDK from Episode 1 — now talking to the REAL gateway.\n"
    "   Open http://127.0.0.1:8000/dashboard : your call is there, with cost, "
    "latency, and tokens.\n"
    "   It's open source (github.com/vahid8/llm-gateway) and already in "
    "production behind a real product.\n"
    "   That's the series. Thanks for building it with me. 🎉"
)

"""Episode 3 — calling YOUR gateway.

The payoff: this is the exact OpenAI SDK from Episode 1 — but base_url now
points at OUR gateway instead of a provider. The app doesn't know or care which
provider answers; it just changes the model string. No keys live here.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/03_gateway.py

Then run this:
    uv run python episodes/03_client.py
"""

from openai import OpenAI

# Point the OpenAI SDK at our own gateway. No real key needed — the gateway
# holds the provider keys, not the app.
client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")

question = "In one sentence, what is an LLM?"

# Same call as Episode 1 — only the model string decides the provider now.
for model in ["gemini/gemini-2.5-flash", "groq/llama-3.3-70b-versatile"]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    print(f"{model}:\n  {response.choices[0].message.content}\n")


# ---------------------------------------------------------------------------
# Don't want to host a gateway yourself? A hosted service like OpenRouter does
# the same job — one endpoint, 100+ models, they hold the keys and the billing.
# It's the same OpenAI-compatible idea: the EXACT client above, just pointed at
# their URL instead of ours. (Free key: https://openrouter.ai/keys)
#
#     client = OpenAI(
#         api_key="sk-or-...",                      # your OpenRouter key
#         base_url="https://openrouter.ai/api/v1",  # instead of our gateway
#     )
#     client.chat.completions.create(
#         model="google/gemini-2.5-flash",          # OpenRouter's model names
#         messages=[{"role": "user", "content": question}],
#     )
#
# Your own gateway = control + cost. OpenRouter = convenience. Same shape.
# ---------------------------------------------------------------------------

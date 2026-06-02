"""Episode 4 — streaming from YOUR gateway.

Same OpenAI SDK, same base_url as Episode 3 — the only change is stream=True.
Now we print each token the instant it arrives instead of waiting for the whole
answer. That "typing" effect you see in ChatGPT? This is it.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/04_gateway.py

Then run this:
    uv run python episodes/04_client.py
"""

from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")

stream = client.chat.completions.create(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": "Explain how a large language model writes text one token at a time, then give a numbered 5-step list of what happens for each token."}],
    stream=True,  # the one new line — ask the gateway to stream
)

# Each chunk carries a little piece of the answer in delta.content.
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()

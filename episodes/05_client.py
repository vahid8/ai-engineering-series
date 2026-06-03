"""Episode 5 — see what your calls cost.

The gateway now tracks tokens, cost, and latency for every call. Here we make a
CHEAP call and an EXPENSIVE one — same tiny answer, but the second buries the
question under a big pile of context. Then we ask the gateway for the running
bill at /usage. Watch the input tokens (and the dollars) explode.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/05_gateway.py

Then run this:
    uv run python episodes/05_client.py
"""

import json

import httpx
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")


def ask(content: str, max_tokens: int | None = None):
    return client.chat.completions.create(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )


# 1) cheap — a short prompt
ask("Say hello in one word.")
# The running bill, straight from the gateway.
bill = httpx.get("http://127.0.0.1:8000/usage").json()

# 2) expensive — the SAME question, buried under ~2,500 words of context.
#    The answer is just as short, but you pay for every input token.
big_context = "Here is some background information you should consider. " * 500
ask(big_context + "\n\nNow, say hello in one word.")

# The running bill, straight from the gateway.
bill = httpx.get("http://127.0.0.1:8000/usage").json()
print(json.dumps(bill, indent=2))

# 3) "Can't I just cap it with max_tokens?"  Yes — but it only caps the OUTPUT.
prompt = "Briefly explain what a CPU does in a computer."
full = ask(prompt)                    # no cap: the model writes the whole answer
capped = ask(prompt, max_tokens=50)   # cap: same prompt, output cut short

print("\nno cap:", full.usage.completion_tokens, "output tokens  ->",
      "finish_reason =", full.choices[0].finish_reason)        # "stop": finished
print("capped:", capped.usage.completion_tokens, "output tokens  ->",
     "finish_reason =", capped.choices[0].finish_reason)      # "length": truncated

# max_tokens shrank the OUTPUT bill — a useful safety cap. But the INPUT (your
# prompt + all that context) is billed in full no matter what; that's the
# explosion above, and max_tokens can't touch it. And don't try to dodge it by
# asking in parts: a follow-up call has to RE-SEND the context plus part 1, so
# you pay for the same tokens twice. The only real fix is to trim the input.

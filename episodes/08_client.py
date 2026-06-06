"""Episode 8 — a multi-user gateway: API keys, rate limits, budgets.

Same OpenAI SDK, same gateway URL. The ONLY change on the client side: the
api_key is now your GATEWAY's key. The SDK sends it as `Authorization: Bearer
<key>` for you — exactly how you'd talk to OpenAI itself.

We play four callers to exercise the three controls:
  1) alice, a valid key             -> works
  2) an intruder with a bad key     -> 401, rejected before any model runs
  3) carol, flooding the gateway    -> 429 once she's over the rate limit
  4) bob, on a tiny budget          -> 402 once his spend crosses the ceiling
Then /usage prints the running bill per user.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/08_gateway.py

Then run this:
    uv run python episodes/08_client.py
"""

import sys

import httpx
from openai import APIStatusError, OpenAI

# Print the ✅ / ⛔ markers on any OS — Windows consoles default to cp1252, which
# can't encode them. (Harmless on macOS/Linux, which are already UTF-8.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GATEWAY = "http://127.0.0.1:8000/v1"
MODEL = "gemini/gemini-2.5-flash"

# Map each rejection's status code to a human reason (so labels are always truthful).
REASON = {401: "bad key", 402: "budget exceeded", 429: "too many requests"}


def ask(api_key: str, prompt: str = "Say hello in exactly five words.") -> str:
    # api_key is OUR gateway's key now. max_retries=0 so a 429/402 fails FAST
    # instead of the SDK silently retrying with backoff (which would hide the
    # rejection and stall the demo).
    client = OpenAI(api_key=api_key, base_url=GATEWAY, max_retries=0)
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


# 1) VALID KEY — alice is known, so her call goes straight through.
print("1) alice (valid key):")
print("   ✅", ask("sk-demo-alice"))

# 2) BAD KEY — an unknown key is refused with 401 before any model is called.
print("\n2) intruder (unknown key):")
try:
    ask("sk-totally-made-up")
except APIStatusError as e:
    print(f"   ⛔ {e.status_code} — rejected (no model call, nothing spent)")

# 3) RATE LIMIT — carol's app has a bug and floods the gateway (limit is 5 / 30s).
print("\n3) carol floods the gateway (limit 5 / 30s):")
for i in range(1, 8):
    try:
        ask("sk-demo-carol", "ping")
        print(f"   call {i}: ✅ ok")
    except APIStatusError as e:
        print(f"   call {i}: ⛔ {e.status_code} — {REASON.get(e.status_code, 'rejected')}")

# 4) BUDGET — bob has a tiny ceiling; he keeps calling until he's cut off.
print("\n4) bob spends until his budget runs out:")
for i in range(1, 11):
    try:
        ask("sk-demo-bob")
        print(f"   call {i}: ✅ ok")
    except APIStatusError as e:
        print(f"   call {i}: ⛔ {e.status_code} — {REASON.get(e.status_code, 'rejected')}")
        break

# 5) THE BILL — /usage shows requests + dollars per user (the owner's view).
print("\n5) /usage — the running bill per user:")
for user, row in httpx.get("http://127.0.0.1:8000/usage").json().items():
    print(f"   {user:6} {row['requests']:>2} reqs   ${row['cost_usd']:.6f}")

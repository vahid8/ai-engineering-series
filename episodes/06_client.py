"""Episode 6 — reliable JSON from an LLM, in three steps.

Same OpenAI SDK, still pointed at your own gateway. We take a messy sentence and
pull structured data out of it:

  1) The naive way — just ask for JSON in the prompt. The model wraps it in a
     ```json fence (or adds chatter), and json.loads() blows up.
  2) JSON mode — response_format={"type": "json_object"}. Now it's always valid
     JSON you can parse. But nothing guarantees WHICH fields you get.
  3) A schema — hand the SDK a Pydantic model. The shape is pinned, and you get
     a typed object back, already validated. This is the one to use.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/06_gateway.py

Then run this:
    uv run python episodes/06_client.py
"""

import json

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")

MODEL = "gemini/gemini-2.5-flash"
SENTENCE = "Ada Lovelace, 36, is a mathematician from London."


# 1) THE NAIVE WAY — just ask, and hope.
print("1) naive prompt")
raw = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user",
               "content": f"Extract name, age, job as JSON: {SENTENCE}"}],
).choices[0].message.content
print("   model returned:", repr(raw))
try:
    data = json.loads(raw)           # ...usually wrapped in ```json fences -> boom
    print("   parsed:", data)
except json.JSONDecodeError as e:
    print(f"   json.loads() FAILED: {e}")


# 2) JSON MODE — guaranteed-parseable JSON (but the shape is still up to the model).
print("\n2) JSON mode")
raw = client.chat.completions.create(
    model=MODEL,
    response_format={"type": "json_object"},
    messages=[{"role": "user",
               "content": f"Extract name, age, job as JSON: {SENTENCE}"}],
).choices[0].message.content
print("   model returned:", repr(raw))
print("   parsed:", json.loads(raw))  # always works now — no fences, valid JSON


# 3) A SCHEMA — pin the exact shape with a Pydantic model.
class Person(BaseModel):
    name: str
    age: int
    job: str


print("\n3) schema (Pydantic)")
# .parse() turns the model into a JSON schema, sends it through the gateway, and
# parses the reply back into a Person for you — already validated.
person = client.chat.completions.parse(
    model=MODEL,
    response_format=Person,
    messages=[{"role": "user", "content": f"Extract the person: {SENTENCE}"}],
).choices[0].message.parsed

print("   got a Person object:", person)
print("   typed fields ->", person.name, "|", person.age, "|", person.job)
print("   person.age is an int:", isinstance(person.age, int))

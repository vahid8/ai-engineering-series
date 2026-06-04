"""Episode 7 — function calling: let the model run YOUR Python.

Same OpenAI SDK, still pointed at your own gateway. An LLM only writes text — it
can't hit an API or read your private data. So we hand it *tools* and let it
decide which one to call.

We give it two, on purpose:
  • get_weather   — an EXTERNAL tool: it calls a real public weather API.
  • get_order_status — an INTERNAL tool: it reads data that lives only in our app.
The model can't know either answer on its own. It picks the right tool (or both)
based on the question.

The agent loop (this is the whole "agent" idea, minus the buzzword):
  repeat, up to MAX_STEPS times:
    1) send the conversation + the tools we offer
    2) did the model ask to call a tool?
         no  -> it answered. print it and STOP.
         yes -> run each requested function, append the results, and loop again.
The model keeps the loop going until it has everything it needs to answer. The
MAX_STEPS cap is the safety belt: never let a tool loop run (and bill) forever
if the model keeps asking or a tool keeps failing.

Start the gateway first (in another terminal):
    uv run --env-file .env python episodes/07_gateway.py

Then run this:
    uv run python episodes/07_client.py
"""

import json

import httpx
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")

MODEL = "gemini/gemini-2.5-flash"


# --- EXTERNAL tool: real data from the internet (a free, no-key weather API). ---
def get_weather(city: str) -> dict:
    geo = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1}).json()
    if not geo.get("results"):
        return {"error": f"unknown city: {city}"}
    loc = geo["results"][0]
    cur = httpx.get("https://api.open-meteo.com/v1/forecast",
                    params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                            "current": "temperature_2m,weather_code"}).json()["current"]
    sky = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
           45: "foggy", 61: "rainy", 71: "snowy", 95: "thunderstorm"}
    return {"city": loc["name"], "temp_c": cur["temperature_2m"],
            "conditions": sky.get(cur["weather_code"], "unknown")}


# --- INTERNAL tool: data that lives only in OUR app (no internet, our code). ---
ORDERS = {"A1023": "shipped", "A1024": "processing", "A1099": "delivered"}


def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": ORDERS.get(order_id, "not found")}


# Map each tool name to the real function, so we can dispatch what the model picks.
TOOL_FUNCS = {"get_weather": get_weather, "get_order_status": get_order_status}


# --- Describe BOTH tools to the model (JSON Schema for each one's arguments). ---
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city (calls a live weather API).",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string",
                                               "description": "City name, e.g. London"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Look up the status of a customer order in our system.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string",
                                                   "description": "Order id, e.g. A1023"}},
                       "required": ["order_id"]}}},
]

messages = [{"role": "user", "content":
             "What's the weather in Tokyo right now, and what's the status of order A1023?"}]

MAX_STEPS = 5  # safety belt: never let the tool loop run (and bill) forever.

# THE AGENT LOOP — keep going until the model stops asking for tools, or we hit the cap.
for step in range(MAX_STEPS):
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.choices[0].message

    # No tool calls -> the model answered. Print it and break out of the loop.
    if not msg.tool_calls:
        print("\n✅ final answer:", msg.content)
        break

    # The model asked to call one or more tools. Keep its turn in the history...
    messages.append(msg)
    # ...then run each requested function and feed the result back.
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = TOOL_FUNCS[tc.function.name](**args)
        print(f"🔧 model called {tc.function.name}({args}) -> {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(result)})
    # loop again: send the tool results back so the model can use them.
else:
    # for/else: this runs only if we never hit `break` — i.e. we used up every
    # step and the model was STILL asking for tools. Bail out instead of looping.
    print(f"\n⚠️  gave up after {MAX_STEPS} tool-calling rounds — no final answer.")

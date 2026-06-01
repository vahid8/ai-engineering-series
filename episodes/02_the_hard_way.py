"""Episode 2 — two providers, the hard way.

We add a second model (Groq's Llama) next to Gemini. Same OpenAI-SDK trick from
Episode 1 — just a different base_url and key. It works... but watch how the
friction piles up: two clients, two base_urls, two key names, and a manual
dispatch that grows an `if` branch for every new provider.

That friction is the whole point of this episode. `02_litellm.py` fixes it.
"""

import os
from pathlib import Path

from openai import OpenAI


def get_key(env_name: str) -> str:
    """Read a key from <ENV_NAME>, or from the file <ENV_NAME>_FILE points to."""
    if key := os.environ.get(env_name):
        return key
    if key_file := os.environ.get(f"{env_name}_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    raise RuntimeError(f"Set {env_name} or {env_name}_FILE in your .env")


# One client per provider — each needs its own base_url and key.
gemini = OpenAI(
    api_key=get_key("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
groq = OpenAI(
    api_key=get_key("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def ask(provider: str, question: str) -> str:
    """Manual dispatch: every new provider means another branch in here."""
    if provider == "gemini":
        client, model = gemini, "gemini-2.5-flash"
    elif provider == "groq":
        client, model = groq, "llama-3.3-70b-versatile"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


question = "In one sentence, what is an LLM?"
print("GEMINI:", ask("gemini", question))
print("GROQ:  ", ask("groq", question))

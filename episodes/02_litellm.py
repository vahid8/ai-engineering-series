"""Episode 2 — the same two providers, with LiteLLM.

One call signature for every provider. To switch models you change ONE string —
the provider is just the prefix ("gemini/...", "groq/..."). Keys are read from
the environment automatically, and you get fallbacks for free.

Compare this to 02_the_hard_way.py: no per-provider clients, no base_urls, no
manual dispatch.
"""

import logging
import os
from pathlib import Path

from litellm import completion, completion_with_fallbacks

logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # quiet LiteLLM's startup warnings


def load_key(env_name: str) -> None:
    """LiteLLM reads keys straight from the environment. We keep ours in files
    (via <ENV_NAME>_FILE), so copy them into the environment once, here."""
    if os.environ.get(env_name):
        return
    if key_file := os.environ.get(f"{env_name}_FILE"):
        os.environ[env_name] = Path(key_file).expanduser().read_text().strip()


load_key("GEMINI_API_KEY")
load_key("GROQ_API_KEY")


def ask(model: str, question: str) -> str:
    response = completion(model=model, messages=[{"role": "user", "content": question}])
    return response.choices[0].message.content


question = "In one sentence, what is an LLM?"

# Same function — only the model string changes. The provider is the prefix.
print("GEMINI:", ask("gemini/gemini-2.5-flash", question))
print("GROQ:  ", ask("groq/llama-3.3-70b-versatile", question))

# Bonus: automatic fallback. If the first model fails, LiteLLM tries the next.
response = completion_with_fallbacks(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": question}],
    fallbacks=["groq/llama-3.3-70b-versatile"],
)
print("FALLBACK:", response.choices[0].message.content)

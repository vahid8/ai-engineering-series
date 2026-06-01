# AI Engineering — from scratch

Companion code for the YouTube series. Free to follow with a Gemini API key.

## Prerequisites
- **uv** — the Python package manager. Install it once:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **A Gemini API key** — get one free at https://aistudio.google.com/apikey

## Setup
```bash
# 1. Clone the repo
git clone https://github.com/vahid8/ai-engineering-series.git
cd ai-engineering-series

# 2. Install dependencies (exact versions, from the lockfile)
uv sync

# 3. Add your Gemini key
cp .env.template .env          # then open .env and paste your key

# 4. Run the first call
uv run --env-file .env first_call.py
```

> Use `uv sync`, **not** `uv init` — `sync` installs the pinned dependencies from
> the committed lockfile. `init` would scaffold a brand-new empty project.

See `SECURITY.md` for the keep-your-keys-safe rules.

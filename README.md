# AI Engineering — from scratch

Companion code for the YouTube series. Free to follow with a Gemini API key.

## Setup
```bash
uv sync                       # install deps from the lockfile
cp .env.template .env         # add your Gemini key
uv run --env-file .env first_call.py
```

See `SECURITY.md` for the keep-your-keys-safe rules.

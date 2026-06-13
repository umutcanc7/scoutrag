# ScoutRAG — shareable demo (your own web UI)

The demo uses **your real ScoutRAG web UI** (`docs/index.html`, the dark-green chat
interface). The FastAPI backend now serves both the UI **and** the API from one server,
so a single public link gives the instructor the full interface — no install on their
side.

Architecture: `python -m uvicorn backend.app:app` serves the UI at `/` and the API at
`/search` + `/health`. A tunnel (cloudflared) exposes that one local port as a public
`https://...` URL.

## One-time setup

```bash
brew install cloudflared      # the tunnel (no account needed)
pip install -r requirements.txt
```

## Run the demo

In **terminal 1** — start the server:

```bash
cd "CS 455 Project"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Wait for `[backend] ready: 40,688 players | semantic=on | ...`.

In **terminal 2** — open the public tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

cloudflared prints a line like:

```
https://random-words-1234.trycloudflare.com   <-- send THIS to the instructor
```

Opening that link shows your full ScoutRAG UI. The frontend auto-detects the tunnel
origin, so it calls the API through the same link (no "Backend URL" setting needed).

The link works **only while both terminals are running** on your computer. `Ctrl+C`
stops them and the link dies. Keep both open during the demo.

> Local-only check first: open `http://localhost:8000` in your own browser to confirm
> everything works before sharing the tunnel link.

## Enabling the local LLM (Ollama) — gets `llm` from off → on

Both the UI and the engine show `llm off` when no local Ollama server is running. The LLM
is **optional** (rule-based fallbacks cover routing, query rewriting and reports, and the
grounding-verification step always runs), but turning it on makes query rewriting and the
scouting reports noticeably better.

```bash
brew install ollama          # or download from ollama.com
ollama serve                 # terminal 3 — starts the server on :11434 (keep running)
ollama pull llama3.1         # one-time model download (~4.7 GB)
```

Then restart the backend — it auto-detects Ollama and `llm` flips to **on**.

Lighter machine? Use a smaller model and point ScoutRAG at it:

```bash
ollama pull llama3.2         # ~2 GB (3B model)
SCOUTRAG_MODEL=llama3.2 python -m uvicorn backend.app:app --port 8000
```

(`SCOUTRAG_MODEL` overrides the default `llama3.1:latest` in `scoutrag/llm.py`.)

Check it's up: `curl http://localhost:11434/api/tags` should return JSON. The status pill
in the UI header will read `llm on`.

## Alternatives

- **Gradio app** (`app_gradio.py`): a simpler, generic UI that prints its own
  `*.gradio.live` link in one command — handy as a backup, but it is not your custom UI.
- **Permanent hosting** (laptop can be off): deploy to a free Hugging Face Space. Note that
  free Spaces cannot run Ollama, so the LLM features would use the rule-based fallback there.
  Ask and I'll prepare the Space layout.

## Data honesty note

The UI states up front that FM24 ratings are designer-set *game* attributes, not real
scouting data — consistent with the instructor's feedback.

"""ScoutRAG local backend (FastAPI).

Runs the full hybrid retrieval engine locally and exposes a small JSON API that
the static web UI (docs/index.html, hostable on GitHub Pages) calls.

Run:
    cd "CS 455 Project"
    pip install -r requirements.txt
    uvicorn backend.app:app --reload --port 8000

Then open docs/index.html (locally or via GitHub Pages) and point it at
http://localhost:8000.

Everything is local. The optional descriptive-term -> constraint rewriting and
the scouting report use a local Ollama model if one is running; otherwise the
rule-based parser and a deterministic template are used.
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoutrag import (load_and_clean, build_profiles, align_embeddings,
                      parse_query, search, NO_MATCH_MESSAGE)
from scoutrag import llm as scout_llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "fmdata24llm.csv")
PKL_PATH = os.path.join(ROOT, "player_embeddings.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="ScoutRAG API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Lazy global state (loaded once at startup)
# --------------------------------------------------------------------------- #
STATE: dict = {"df": None, "emb": None, "embed_fn": None}


def _build_embed_fn():
    """Return a function text -> 384-d unit vector, or None if model missing."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)

        def embed_query(text):
            return model.encode([text], normalize_embeddings=True)[0].astype("float32")
        return embed_query
    except Exception as e:  # pragma: no cover
        print(f"[backend] semantic search disabled (no sentence-transformers): {e}")
        return None


@app.on_event("startup")
def _startup():
    print("[backend] loading dataset ...")
    df = build_profiles(load_and_clean(CSV_PATH, verbose=True))
    STATE["df"] = df
    STATE["emb"] = align_embeddings(df, PKL_PATH)
    STATE["embed_fn"] = _build_embed_fn() if STATE["emb"] is not None else None
    print(f"[backend] ready: {len(df):,} players | "
          f"semantic={'on' if STATE['embed_fn'] else 'off'} | "
          f"ollama={'on' if scout_llm.ollama_available() else 'off'}")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    use_llm: bool = True       # use Ollama to rewrite descriptive queries
    make_report: bool = False  # also generate a grounded scouting report


@app.get("/health")
def health():
    return {
        "status": "ok",
        "players": 0 if STATE["df"] is None else int(len(STATE["df"])),
        "semantic": STATE["embed_fn"] is not None,
        "ollama": scout_llm.ollama_available(),
    }


@app.post("/search")
def do_search(req: SearchRequest):
    df = STATE["df"]
    query = req.query.strip()

    rewritten, used_llm = (query, False)
    if req.use_llm:
        rewritten, used_llm = scout_llm.rewrite_query(query)

    parsed = parse_query(rewritten)
    # The count written in the query ("top 5") wins; otherwise use the UI default.
    if req.top_k and not parsed.get("top_k_explicit"):
        parsed["top_k"] = req.top_k

    result = search(df, parsed, embeddings=STATE["emb"], embed_query_fn=STATE["embed_fn"])

    payload = {
        "query": query,
        "rewritten_query": rewritten if used_llm else None,
        "used_llm": used_llm,
        "message": result["message"],
        "players": result["players"],
        "n_filtered": result["n_filtered"],
        "constraints": parsed["trace"],
        "summary": scout_llm.generate_summary(query, result["players"]) if result["players"] else None,
    }

    if req.make_report and result["players"]:
        report = scout_llm.generate_report(query, result["players"])
        payload["report"] = report
        payload["grounding"] = scout_llm.verify_grounding(report, result["players"])

    return payload

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
                      parse_query, search, structured_filter, find_players_by_name,
                      find_similar_players, players_to_records,
                      has_constraints, ATTR_READABLE, NO_MATCH_MESSAGE)
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
STATE: dict = {"df": None, "emb": None, "embed_fn": None, "clubs": None}


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


def _load_or_build_embeddings(df):
    """Load the cached play-style embeddings; if the cache is missing, build it
    once (and save it for next time) so semantic search works after the cache is
    deleted/changed -- no separate build_embeddings.py run needed."""
    emb = align_embeddings(df, PKL_PATH)
    if emb is not None:
        return emb
    try:
        import numpy as np
        import pickle
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # sentence-transformers not installed -> stay off
        print(f"[backend] no embeddings cache and cannot build it: {e}")
        return None
    print("[backend] no embeddings cache found -- building once (this takes a "
          "couple of minutes, downloads MiniLM the first time) ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = model.encode(df["embed_text"].tolist(), batch_size=64,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype("float32")
    # Save a raw-CSV-indexed cache (what align_embeddings expects next startup).
    try:
        dim = vecs.shape[1]
        n_raw = int(df["raw_idx"].max()) + 1
        full = np.zeros((n_raw, dim), dtype="float32")
        full[df["raw_idx"].to_numpy()] = vecs
        with open(PKL_PATH, "wb") as f:
            pickle.dump({"embeddings": full, "model": EMBED_MODEL_NAME, "dim": dim}, f)
        print(f"[backend] saved embeddings cache -> {PKL_PATH}")
    except Exception as e:
        print(f"[backend] built embeddings but could not save cache: {e}")
    return vecs  # already in cleaned-df order, ready to use


@app.on_event("startup")
def _startup():
    print("[backend] loading dataset ...")
    df = build_profiles(load_and_clean(CSV_PATH, verbose=True))
    STATE["df"] = df
    STATE["emb"] = _load_or_build_embeddings(df)
    STATE["embed_fn"] = _build_embed_fn() if STATE["emb"] is not None else None
    # distinct, real club names (drop the '-' placeholder) for club-name matching
    STATE["clubs"] = sorted({c for c in df["Club_Name"].dropna().unique()
                             if isinstance(c, str) and c.strip() not in ("", "-")})
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
    clubs = STATE["clubs"]
    query = req.query.strip()

    # --- 1. Route the query: info / similar / search ------------------------ #
    # Every query goes through the router. The LLM decides the intent when
    # Ollama is running; otherwise a rule-based fallback is used.
    route = scout_llm.route_query(query) if req.use_llm else scout_llm._heuristic_route(query)
    intent = route["intent"]

    # --- 2. INFO: a single named player ------------------------------------- #
    if intent == "info":
        hits = find_players_by_name(df, route["player"] or query, limit=5)
        if len(hits):
            hits = hits.assign(match_score=(hits["Overall"].fillna(0) / 20.0).round(3))
            records = players_to_records(hits)
            return {
                "query": query, "mode": "info",
                "rewritten_query": None, "used_llm": route["used_llm"],
                "message": None, "players": records, "n_filtered": len(records),
                "constraints": [f"player lookup: {records[0]['Player_Name']}"],
                "summary": scout_llm.generate_player_info(records[0]),
            }
        # fall through to search if the name wasn't found

    # --- 3. SIMILAR: players like a reference player ------------------------ #
    if intent == "similar":
        ref = find_players_by_name(df, route["player"] or query, limit=1)
        if len(ref):
            k = route["top_k"] or req.top_k or 5
            ref_name = str(ref.iloc[0]["Player_Name"])

            # Honour any EXTRA constraints in the query (nationality, age, club,
            # foot, value...) so "Turkish players like Haaland" stays Turkish.
            # Parse the query with the reference name stripped so the player's
            # own name can't be read as a constraint.
            residual = query.lower().replace(ref_name.lower(), " ")
            extra = parse_query(residual, known_clubs=clubs)
            allowed_index, extra_trace = None, []
            if has_constraints(extra):
                filtered, applied, _ = structured_filter(df, extra)
                allowed_index = filtered.index
                extra_trace = applied

            sims = find_similar_players(df, ref, embeddings=STATE["emb"],
                                        top_k=k, allowed_index=allowed_index)
            records = players_to_records(sims)
            constraints = [f"similar to: {ref_name}",
                           "by embedding" if STATE["emb"] is not None else "by attributes"]
            constraints += extra_trace
            return {
                "query": query, "mode": "similar",
                "rewritten_query": None, "used_llm": route["used_llm"],
                "message": None if records else NO_MATCH_MESSAGE,
                "players": records, "n_filtered": len(records),
                "constraints": constraints,
                "summary": scout_llm.generate_summary(query, records) if records else None,
            }
        # fall through to search if the reference name wasn't found

    # --- 4. SEARCH: constrained hybrid retrieval ---------------------------- #
    # The ORIGINAL user query is the authoritative source of constraints. We parse
    # it deterministically so the rule-based parser controls every hard filter
    # (foot, age, value, wage, position, nationality, club, explicit numeric attrs).
    # The LLM rewrite is used ONLY to fill in a position / nationality / club that
    # the deterministic parser missed, and to reinforce a soft signal for an
    # attribute the user actually named. The LLM can never invent a numeric
    # threshold or override age/foot/value/wage -> no input-side hallucinated
    # constraints, and the user's own words are never dropped.
    rewritten = route["query"] or query
    used_llm = route["used_llm"] and rewritten.strip().lower() != query.strip().lower()

    parsed = parse_query(query, known_clubs=clubs)

    if used_llm:
        extra = parse_query(rewritten, known_clubs=clubs)
        for fld in ("position", "nationality", "club"):
            if not parsed.get(fld) and extra.get(fld):
                parsed[fld] = extra[fld]
                parsed["trace"].append(f"{fld} (LLM-assisted) = {extra[fld]}")
        ql = query.lower()
        for col, direction in extra.get("soft_attrs", {}).items():
            named = ATTR_READABLE.get(col, col).lower() in ql
            if named and col not in parsed["hard_attrs"] and col not in parsed["soft_attrs"]:
                parsed["soft_attrs"][col] = direction
                parsed["trace"].append(f"soft (LLM-assisted): {ATTR_READABLE.get(col, col)}{direction}")

    # Safety net: a bare player name that carried no constraints (common when the
    # LLM router is off) should still return an info card rather than a generic
    # quality-ranked list.
    if not has_constraints(parsed):
        hits = find_players_by_name(df, query, limit=5)
        if len(hits):
            hits = hits.assign(match_score=(hits["Overall"].fillna(0) / 20.0).round(3))
            records = players_to_records(hits)
            return {
                "query": query, "mode": "info",
                "rewritten_query": None, "used_llm": route["used_llm"],
                "message": None, "players": records, "n_filtered": len(records),
                "constraints": [f"player lookup: {records[0]['Player_Name']}"],
                "summary": scout_llm.generate_player_info(records[0]),
            }

    # explicit "top N" in the query wins; then the router's count; then the UI default
    if not parsed.get("top_k_explicit"):
        if route["top_k"]:
            parsed["top_k"] = route["top_k"]
        elif req.top_k:
            parsed["top_k"] = req.top_k

    result = search(df, parsed, embeddings=STATE["emb"], embed_query_fn=STATE["embed_fn"])

    payload = {
        "query": query, "mode": "search",
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

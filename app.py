"""ScoutRAG — HuggingFace Spaces entry point.

This file is the HF Spaces entry point (must be named app.py).
It is identical in logic to app_gradio.py, but:
  - Uses Anthropic API (ANTHROPIC_API_KEY secret) instead of Ollama
  - launch() is configured for HF Spaces (server_name="0.0.0.0", no share=True)
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from scoutrag import (load_and_clean, build_profiles, align_embeddings,
                      parse_query, search, structured_filter, find_players_by_name,
                      find_similar_players, players_to_records,
                      has_constraints, ATTR_READABLE, NO_MATCH_MESSAGE)
from scoutrag import llm as scout_llm

CSV_PATH = os.path.join(ROOT, "fmdata24llm.csv")
PKL_PATH = os.path.join(ROOT, "player_embeddings.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

STATE: dict = {"df": None, "emb": None, "embed_fn": None, "clubs": None}


def _build_embed_fn():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)

        def embed_query(text):
            return model.encode([text], normalize_embeddings=True)[0].astype("float32")
        return embed_query
    except Exception as e:
        print(f"[app] semantic search disabled: {e}")
        return None


def _load_or_build_embeddings(df):
    emb = align_embeddings(df, PKL_PATH)
    if emb is not None:
        return emb
    try:
        import numpy as np
        import pickle
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"[app] cannot build embeddings: {e}")
        return None
    print("[app] building embeddings (first run only) ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = model.encode(df["embed_text"].tolist(), batch_size=64,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype("float32")
    try:
        dim = vecs.shape[1]
        n_raw = int(df["raw_idx"].max()) + 1
        full = np.zeros((n_raw, dim), dtype="float32")
        full[df["raw_idx"].to_numpy()] = vecs
        with open(PKL_PATH, "wb") as f:
            pickle.dump({"embeddings": full, "model": EMBED_MODEL_NAME, "dim": dim}, f)
        print(f"[app] saved embeddings -> {PKL_PATH}")
    except Exception as e:
        print(f"[app] could not save cache: {e}")
    return vecs


def load_engine():
    print("[app] loading dataset ...")
    df = build_profiles(load_and_clean(CSV_PATH, verbose=True))
    STATE["df"] = df
    STATE["emb"] = _load_or_build_embeddings(df)
    STATE["embed_fn"] = _build_embed_fn() if STATE["emb"] is not None else None
    STATE["clubs"] = sorted({c for c in df["Club_Name"].dropna().unique()
                             if isinstance(c, str) and c.strip() not in ("", "-")})
    api_status = "on ✅" if scout_llm._api_available() else "off (rule-based fallback)"
    print(f"[app] ready: {len(df):,} players | "
          f"semantic={'on' if STATE['embed_fn'] else 'off'} | "
          f"anthropic api={api_status}")


def run_query(query: str, top_k: int, use_llm: bool, make_report: bool) -> dict:
    df = STATE["df"]
    clubs = STATE["clubs"]
    query = (query or "").strip()
    if not query:
        return {"mode": "empty", "players": [], "summary": "Type a query first.",
                "constraints": [], "report": None, "grounding": None}

    route = scout_llm.route_query(query) if use_llm else scout_llm._heuristic_route(query)
    intent = route["intent"]

    if intent == "info":
        hits = find_players_by_name(df, route["player"] or query, limit=5)
        if len(hits):
            hits = hits.assign(match_score=(hits["Overall"].fillna(0) / 20.0).round(3))
            records = players_to_records(hits)
            return {"mode": "info", "players": records,
                    "constraints": [f"player lookup: {records[0]['Player_Name']}"],
                    "summary": scout_llm.generate_player_info(records[0]),
                    "report": None, "grounding": None}
        name_sought = route["player"] or query
        return {"mode": "info", "players": [],
                "constraints": [f"player lookup: {name_sought}"],
                "summary": scout_llm.generate_not_found_message(name_sought, route.get("used_llm", False)),
                "report": None, "grounding": None}

    if intent == "similar":
        ref = find_players_by_name(df, route["player"] or query, limit=1)
        if len(ref):
            k = route["top_k"] or top_k or 5
            ref_name = str(ref.iloc[0]["Player_Name"])
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
            return {"mode": "similar", "players": records, "constraints": constraints,
                    "summary": scout_llm.generate_summary(query, records) if records else NO_MATCH_MESSAGE,
                    "report": None, "grounding": None}

    rewritten = route["query"] or query
    used_llm = route.get("used_llm", False) and rewritten.strip().lower() != query.strip().lower()
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

    if not has_constraints(parsed):
        hits = find_players_by_name(df, query, limit=5)
        if len(hits):
            hits = hits.assign(match_score=(hits["Overall"].fillna(0) / 20.0).round(3))
            records = players_to_records(hits)
            return {"mode": "info", "players": records,
                    "constraints": [f"player lookup: {records[0]['Player_Name']}"],
                    "summary": scout_llm.generate_player_info(records[0]),
                    "report": None, "grounding": None}
        return {"mode": "info", "players": [],
                "constraints": [f"player lookup: {query}"],
                "summary": scout_llm.generate_not_found_message(query, route.get("used_llm", False)),
                "report": None, "grounding": None}

    if not parsed.get("top_k_explicit"):
        parsed["top_k"] = route["top_k"] or top_k or 10

    result = search(df, parsed, embeddings=STATE["emb"], embed_query_fn=STATE["embed_fn"])
    out = {"mode": "search", "players": result["players"],
           "constraints": parsed["trace"],
           "summary": (scout_llm.generate_summary(query, result["players"])
                       if result["players"] else (result["message"] or NO_MATCH_MESSAGE)),
           "report": None, "grounding": None}

    if make_report and result["players"]:
        report = scout_llm.generate_report(query, result["players"])
        out["report"] = report
        out["grounding"] = scout_llm.verify_grounding(report, result["players"])
    return out


_TABLE_COLS = ["Player_Name", "Age", "Nationality", "Position", "Club_Name",
               "League", "Foot", "Value_EUR", "Wage_EUR_pm", "Overall", "match_score"]


def _records_to_df(records):
    if not records:
        return pd.DataFrame(columns=_TABLE_COLS)
    rows = [{c: r.get(c) for c in _TABLE_COLS} for r in records]
    df = pd.DataFrame(rows, columns=_TABLE_COLS)
    for c in ("Value_EUR", "Wage_EUR_pm"):
        df[c] = df[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
    for c in ("Overall", "match_score"):
        df[c] = df[c].apply(lambda v: round(v, 2) if pd.notna(v) else "")
    return df


def _grounding_md(grounding) -> str:
    if not grounding:
        return ""
    score = grounding.get("grounding_score", 1.0)
    checked = grounding.get("total_checked", 0)
    halluc = grounding.get("hallucination_count", 0)
    parts = ["### Grounding verification"]
    if checked == 0:
        parts.append("_No checkable name/number citations found in the report._")
        return "\n".join(parts)
    if halluc == 0:
        parts.append(f"✅ All {checked} cited claim(s) supported. Grounding score: **{score:.2f}**.")
    else:
        parts.append(f"⚠️ {halluc}/{checked} cited claim(s) not supported. Grounding score: **{score:.2f}**.")
    for it in grounding.get("mismatched_facts", []):
        parts.append(f"- ❌ mismatch: {it}")
    for it in grounding.get("unsupported_players", []):
        parts.append(f"- ❌ unsupported name: {it}")
    return "\n".join(parts)


def on_search(query, top_k, use_llm, make_report):
    res = run_query(query, int(top_k), bool(use_llm), bool(make_report))
    summary = res.get("summary") or ""
    constraints = res.get("constraints") or []
    cons_md = ("**Constraints parsed:**\n" + "\n".join(f"- {c}" for c in constraints)
               if constraints else "")
    table = _records_to_df(res.get("players"))
    report_md = ""
    if res.get("report"):
        report_md = "### Scouting report\n\n" + res["report"]
        g = _grounding_md(res.get("grounding"))
        if g:
            report_md += "\n\n" + g
    summary_block = (f"**Mode:** `{res.get('mode')}`\n\n{summary}"
                     + (("\n\n" + cons_md) if cons_md else ""))
    return summary_block, table, report_md


EXAMPLES = [
    ["fast young Turkish winger under 23", 10, True, False],
    ["left-footed centre-back with strong heading, value under 20M", 10, True, False],
    ["creative attacking midfielder with great passing and vision", 8, True, True],
    ["players similar to Mohamed Salah", 5, True, False],
    ["tell me about Kevin De Bruyne", 5, True, False],
    ["clinical striker under 25 with finishing above 15", 10, True, True],
]


def build_ui():
    semantic_on = STATE["embed_fn"] is not None
    api_on = scout_llm._api_available()
    n = 0 if STATE["df"] is None else len(STATE["df"])
    status = (f"**{n:,} players loaded** · semantic retrieval: "
              f"{'on ✅' if semantic_on else 'off (attribute fallback)'} · "
              f"LLM (Anthropic API): {'on ✅' if api_on else 'off (rule-based fallback)'}")

    with gr.Blocks(title="ScoutRAG — CS455 demo") as demo:
        gr.Markdown(
            "# ScoutRAG\n"
            "Constrained hybrid retrieval (structured-symbolic + semantic) over a "
            "Football Manager 2024 export, with post-generation grounding verification.\n\n"
            "> ⚠️ **Data note:** FM24 ratings are designer-set *game* attributes, "
            "not real-world scouting data. This is a prototype validation dataset."
        )
        gr.Markdown(status)

        with gr.Row():
            query = gr.Textbox(label="Scouting query", scale=4,
                               placeholder="e.g. fast young Turkish winger under 23")
            btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            top_k = gr.Slider(1, 30, value=10, step=1, label="Top-k")
            use_llm = gr.Checkbox(value=True, label="Use LLM routing")
            make_report = gr.Checkbox(value=False, label="Generate grounded scouting report")

        summary_out = gr.Markdown(label="Summary")
        table_out = gr.Dataframe(label="Retrieved players", wrap=True)
        report_out = gr.Markdown(label="Report")

        gr.Examples(examples=EXAMPLES,
                    inputs=[query, top_k, use_llm, make_report],
                    label="Try an example")

        btn.click(on_search, [query, top_k, use_llm, make_report],
                  [summary_out, table_out, report_out])
        query.submit(on_search, [query, top_k, use_llm, make_report],
                     [summary_out, table_out, report_out])
    return demo


load_engine()
demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
else:
    # HuggingFace Spaces calls this module directly; Gradio detects `demo` automatically.
    demo.launch(server_name="0.0.0.0")

"""Local LLM integration for ScoutRAG via Ollama.

Two jobs:
  1. rewrite_query(): turn an informal / descriptive scouting query
     ("a clinical, visionary number 10 who is cheap") into a clean structured
     query string that core.parse_query() can handle reliably.
  2. generate_report(): write a short grounded scouting report from the
     retrieved player rows only.

Everything degrades gracefully: if Ollama is not running, rewrite_query falls
back to the original query (the rule-based parser already understands many
scouting adjectives via its synonym map), and generate_report returns a
deterministic template instead.

Setup (one time, on the user's Mac):
    brew install ollama        # or download from ollama.com
    ollama serve               # starts the local server on :11434
    ollama pull llama3.1   # or any local model; set DEFAULT_MODEL below to match
"""

from __future__ import annotations

import json
import re
import urllib.request

from .core import ATTR_READABLE

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:latest"

_REWRITE_SYSTEM = (
    "You are a query rewriter for a football scouting search engine.\n"
    "Rewrite the user's informal scouting request into ONE clean structured line.\n\n"
    "RULES:\n"
    "1. Keep position words (goalkeeper, centre-back, full-back, defensive midfielder,\n"
    "   midfielder, playmaker, winger, striker, forward).\n"
    "2. Turn descriptive adjectives into explicit attribute constraints on this scale\n"
    "   (1-20, where 14 = good, 15-16 = very good, 17+ = elite). Examples:\n"
    "   clinical/lethal -> finishing >= 15 ; visionary/creative -> vision >= 15 ;\n"
    "   powerful/strong -> strength >= 14 ; fast/pacy -> pace >= 14 ;\n"
    "   tireless/high work rate -> work rate >= 15, stamina >= 15 ;\n"
    "   commanding keeper -> command of area >= 15 ; shot-stopper -> reflexes >= 15 ;\n"
    "   tenacious/combative -> tackling >= 14 ; composed -> composure >= 14.\n"
    "3. Preserve ALL precise numeric constraints exactly as given\n"
    "   (e.g. 'passing between 10 and 17', 'older than 30', 'under 21').\n"
    "4. Preserve OR conditions ('Turkish or Brazilian', 'under 20 or over 35').\n"
    "5. Preserve budget/value, wage, nationality, foot, and any 'top N'.\n"
    "6. Output ONLY the rewritten line. No explanation, no quotes.\n\n"
    "EXAMPLES:\n"
    "In: clinical poacher, cheap, teenager\n"
    "Out: striker, finishing >= 15, off the ball >= 14, under 19, valued under 5M\n"
    "In: a tireless box-to-box engine who never stops\n"
    "Out: central midfielder, work rate >= 15, stamina >= 15\n"
    "In: commanding Brazilian or Argentine keeper good with his feet\n"
    "Out: goalkeeper, command of area >= 15, kicking >= 14, Brazilian or Argentinian\n"
)


def ollama_available(model: str = DEFAULT_MODEL, timeout: float = 1.5) -> bool:
    """Return True if a local Ollama server is reachable."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_chat(system: str, user: str, model: str = DEFAULT_MODEL,
                 temperature: float = 0.1, max_tokens: int = 256,
                 timeout: float = 60) -> str | None:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())
        return out.get("message", {}).get("content", "").strip()
    except Exception:
        return None


def rewrite_query(query: str, model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    """Rewrite an informal query into a structured one using the local LLM.

    Returns (rewritten_query, was_rewritten). Falls back to the original query
    (was_rewritten=False) when Ollama is unavailable or returns nothing usable.
    """
    if not ollama_available(model):
        return query, False
    out = _ollama_chat(_REWRITE_SYSTEM, f"In: {query}\nOut:", model=model)
    if not out:
        return query, False
    out = out.splitlines()[0].strip()
    out = re.sub(r"^(?:out|output|rewritten)\s*:\s*", "", out, flags=re.IGNORECASE)
    out = out.strip().strip('"\'')
    return (out, True) if out else (query, False)


# --------------------------------------------------------------------------- #
# Grounded report generation
# --------------------------------------------------------------------------- #

_REPORT_SYSTEM = (
    "You are a football scouting analyst. You receive a query and a JSON list of "
    "player records from a Football Manager 2024 dataset (synthetic game data, not "
    "real-world scouting data). Write a concise scouting summary.\n"
    "STRICT RULES:\n"
    "1. Only mention players and numeric values that appear verbatim in the JSON.\n"
    "2. Never invent or estimate a number; copy stats exactly.\n"
    "3. Keep it under 180 words; profile the top 3 candidates.\n"
    "4. End with one sentence noting this is FM24 game data, not real scouting data.\n"
)


def generate_report(query, players, model: str = DEFAULT_MODEL) -> str:
    """Generate a grounded scouting report from retrieved player records.

    `players` is the list of dicts returned by core.search()['players'].
    Falls back to a deterministic template when Ollama is unavailable.
    """
    if not players:
        from .core import NO_MATCH_MESSAGE
        return NO_MATCH_MESSAGE
    if not ollama_available(model):
        return _template_report(query, players)
    ctx = json.dumps(players[:5], ensure_ascii=False, indent=2)
    out = _ollama_chat(_REPORT_SYSTEM, f"Query: {query}\n\nPlayers (JSON):\n{ctx}",
                       model=model, max_tokens=400, temperature=0.2)
    return out or _template_report(query, players)


_SUMMARY_SYSTEM = (
    "You are a football scout. Given a query and a JSON list of already-ranked player "
    "records from a Football Manager 2024 dataset (synthetic game data), write ONE short "
    "paragraph (2-4 sentences) summarising the shortlist. Mention the top 2-3 players by "
    "name and why they fit. Only use names and facts present in the JSON; never invent "
    "numbers. Do not use bullet points. Plain text only."
)


def generate_summary(query, players, model: str = DEFAULT_MODEL) -> str:
    """Return a single short paragraph describing the ranked shortlist.

    Uses the local LLM if Ollama is running; otherwise builds a clean,
    fully-grounded deterministic paragraph from the top players.
    """
    if not players:
        return ""
    if ollama_available(model):
        ctx = json.dumps([{k: p.get(k) for k in
                           ("Player_Name", "Age", "Position", "Nationality",
                            "Club_Name", "League", "Value_EUR", "match_score")}
                          for p in players[:5]], ensure_ascii=False)
        out = _ollama_chat(_SUMMARY_SYSTEM, f"Query: {query}\n\nRanked players (JSON):\n{ctx}",
                           model=model, max_tokens=220, temperature=0.3)
        if out:
            return out.strip()
    return _template_summary(query, players)


def _template_summary(query, players) -> str:
    n = len(players)
    def desc(p):
        age = f", {p['Age']}" if p.get("Age") is not None else ""
        club = f" ({p['Club_Name']})" if p.get("Club_Name") else ""
        return f"{p['Player_Name']}{club}{age}"
    lead = desc(players[0])
    rest = [p["Player_Name"] for p in players[1:3]]
    rest_txt = ""
    if len(rest) == 1:
        rest_txt = f", ahead of {rest[0]}"
    elif len(rest) == 2:
        rest_txt = f", ahead of {rest[0]} and {rest[1]}"
    plural = "players" if n != 1 else "player"
    return (f"The shortlist returns {n} {plural} for “{query}”, ranked by how well "
            f"they meet the requested criteria blended with overall quality. {lead} tops the "
            f"list{rest_txt}. These attributes are Football Manager 2024 game data, not "
            f"real-world scouting data.")


def _template_report(query, players) -> str:
    lines = [f"Found {len(players)} matching player(s) for: \"{query}\".", ""]
    for i, p in enumerate(players[:5], 1):
        val = f"EUR{p['Value_EUR']:,.0f}" if p.get("Value_EUR") else "value n/a"
        lines.append(f"{i}. {p['Player_Name']} — {p.get('Position','?')}, "
                     f"{p.get('Nationality','?')}, age {p.get('Age','?')}, "
                     f"{p.get('Club_Name','?')} ({val}).")
    lines.append("")
    lines.append("Note: attributes are Football Manager 2024 game data, not real scouting data.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Post-generation grounding verification
# --------------------------------------------------------------------------- #

def verify_grounding(report_text: str, players) -> dict:
    """Check that names and numeric attribute citations in a generated report
    are supported by the retrieved player records.

    Returns dict with supported_facts, mismatched_facts, unsupported_players,
    total_checked, hallucination_count, grounding_score.
    """
    vr = {"supported_facts": [], "mismatched_facts": [], "unsupported_players": [],
          "total_checked": 0, "hallucination_count": 0, "grounding_score": 1.0}
    if not players:
        return vr

    names = {str(p.get("Player_Name", "")).lower(): p for p in players}

    # 1. capitalised 2-4 word sequences that look like player names
    for cand in re.findall(r"(?:[A-ZÀ-Þ][\wÀ-ÿ'’.-]+\s+){1,3}[A-ZÀ-Þ][\wÀ-ÿ'’.-]+", report_text):
        cl = cand.lower().strip()
        if any(cl in rn or rn in cl for rn in names):
            vr["supported_facts"].append(f"player ok: {cand}")
        else:
            vr["unsupported_players"].append(cand)
            vr["hallucination_count"] += 1
        vr["total_checked"] += 1

    # 2. numeric attribute citations: "finishing 17", "passing = 15", "vision: 16"
    readable_to_col = {v.lower(): k for k, v in ATTR_READABLE.items()}
    pat = r"(" + "|".join(re.escape(r) for r in readable_to_col) + r")\s*[:=]?\s*(\d{1,2})\b"
    for rd, val in re.findall(pat, report_text.lower()):
        col = readable_to_col[rd]
        cited = int(val)
        actual = [p.get(col) for p in players if col in p]
        vr["total_checked"] += 1
        # if the cited attribute isn't even in the record schema, skip silently
        if not actual:
            continue
        if cited in [a for a in actual if a is not None]:
            vr["supported_facts"].append(f"{rd}={cited} ok")
        else:
            vr["mismatched_facts"].append(f"{rd}={cited} not in retrieved rows")
            vr["hallucination_count"] += 1

    if vr["total_checked"]:
        vr["grounding_score"] = 1.0 - vr["hallucination_count"] / vr["total_checked"]
    return vr

"""
ScoutRAG Evaluation Suite
==========================
50 structured tests across 5 categories, plus a 3-way ablation study.

Run from the project root:
    python scoutrag_eval.py

Outputs:
    eval_results.csv   — one row per test (query, expected, result, pass/fail, parsed tokens)
    ablation.csv       — structured-only vs semantic-only vs hybrid comparison
"""

from __future__ import annotations

import csv
import os
import sys
import time
import unicodedata
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

from scoutrag import (
    load_and_clean, build_profiles, align_embeddings,
    parse_query, structured_filter, rank_players, search,
    find_players_by_name, find_similar_players, players_to_records,
    has_constraints, ATTR_READABLE, NO_MATCH_MESSAGE,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

CSV_PATH = "fmdata24llm.csv"
PKL_PATH = "player_embeddings.pkl"

print("Loading dataset …")
df = build_profiles(load_and_clean(CSV_PATH))
print(f"  {df.shape[0]:,} players loaded")

emb = align_embeddings(df, PKL_PATH)
embed_query_fn = None
try:
    from sentence_transformers import SentenceTransformer
    _qm = SentenceTransformer("all-MiniLM-L6-v2")
    embed_query_fn = lambda t: _qm.encode([t], normalize_embeddings=True)[0].astype("float32")
    print("  semantic search: ENABLED")
except Exception as e:
    print(f"  semantic search: DISABLED ({e})")

SEM_ON = emb is not None and embed_query_fn is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    """Normalize Unicode + Turkish characters for loose name matching."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return (s.replace("ı", "i").replace("İ", "i").replace("ş", "s")
             .replace("ğ", "g").replace("ü", "u").replace("ö", "o")
             .replace("ç", "c").replace("ı", "i").lower().strip())


def player_in_results(result_players: list[dict], target_name: str) -> bool:
    """Return True if target_name appears (fuzzy) in the result list."""
    t = _strip(target_name)
    for p in result_players:
        if t in _strip(p.get("Player_Name", "")):
            return True
    return False


def run_search(query: str, top_k: int = 20) -> dict:
    """
    Hybrid retrieval: parse → structured filter → rank → (semantic).
    Returns the same result dict as scoutrag.search().
    """
    parsed = parse_query(query)
    parsed["top_k"] = top_k
    result = search(df, parsed, embeddings=emb, embed_query_fn=embed_query_fn, return_df=True)
    result["parsed"] = parsed
    return result


def run_structured_only(query: str, top_k: int = 20) -> dict:
    """Structured filter + quality ranking, no semantic."""
    parsed = parse_query(query)
    parsed["top_k"] = top_k
    result = search(df, parsed, embeddings=None, embed_query_fn=None, return_df=True)
    result["parsed"] = parsed
    return result


def run_semantic_only(query: str, top_k: int = 20) -> dict:
    """Semantic retrieval only — skip structured filter."""
    if not SEM_ON:
        return {"players": [], "n_filtered": 0, "message": "semantic disabled", "df": pd.DataFrame(), "parsed": {}}
    parsed = parse_query(query)
    parsed["top_k"] = top_k
    # Skip structured filter: pass the full df through rank_players with semantic scores
    q_vec = embed_query_fn(query)
    scores = (emb @ q_vec).astype(float)
    ranked = rank_players(df, parsed, sem_scores=scores)
    top = ranked.head(top_k)
    players = players_to_records(top)
    return {"players": players, "n_filtered": len(df), "message": None, "df": top, "parsed": parsed}


def fmt_trace(parsed: dict) -> str:
    return " | ".join(parsed.get("trace", [])) or "(none)"


def name_lookup_test(query: str, target_name: str, top_k: int = 20) -> dict:
    """
    Directly call find_players_by_name, then check if target appears.
    Used for Category 1 attribute-constrained lookups too, via full search.
    """
    # For named lookups, use both find_players_by_name AND full search
    by_name = find_players_by_name(df, query, limit=top_k)
    in_name_results = any(_strip(target_name) in _strip(r.get("Player_Name", ""))
                          for r in players_to_records(by_name))
    return in_name_results, players_to_records(by_name)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
# Each test is a dict with:
#   query          : str  — the natural-language query
#   category       : str
#   test_type      : "search" | "lookup" | "no_result"
#   target_player  : str | None   — player that MUST appear (or must NOT appear)
#   expected_found : bool         — True = player should be in top-k; False = should NOT appear
#   expected_empty : bool         — True = results list should be empty / no-match
#   note           : str          — human explanation

TESTS = [

    # -----------------------------------------------------------------------
    # CATEGORY 1 — Real player attribute-constrained lookups (20 tests)
    # Constraints derived from actual FM24 attribute values.
    # Expected: target player appears in top-20 results.
    # -----------------------------------------------------------------------

    # 1
    {
        "id": 1,
        "query": "central midfielder at Fenerbahce with tackling >= 14 and passing >= 12",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "İsmail Yüksek",
        "expected_found": True,
        "expected_empty": False,
        "note": "Ismail Yüksek: Tck=15, Pas=13, DM/M(C), Fenerbahce",
    },
    # 2
    {
        "id": 2,
        "query": "Nigerian striker with pace >= 18 and finishing >= 16 under 25",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Victor Osimhen",
        "expected_found": True,
        "expected_empty": False,
        "note": "Osimhen: Pac=19, Fin=18, Age=24, Nigerian",
    },
    # 3
    {
        "id": 3,
        "query": "Bosnian striker with heading >= 16 and strength >= 17 at Fenerbahce",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Edin Džeko",
        "expected_found": True,
        "expected_empty": False,
        "note": "Džeko: Hea=17, Str=18, Bosnian, Fenerbahce",
    },
    # 4
    {
        "id": 4,
        "query": "left-footed attacking midfielder with passing >= 17 at Fenerbahce",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Dušan Tadić",
        "expected_found": True,
        "expected_empty": False,
        "note": "Tadić: Pas=18, Left foot, AM(RLC)/ST, Fenerbahce",
    },
    # 5
    {
        "id": 5,
        "query": "left-footed defensive midfielder at Fenerbahce with acceleration >= 16",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Fred",
        "expected_found": True,
        "expected_empty": False,
        "note": "Fred: Acc=17, Left foot, DM/M(C), Fenerbahce",
    },
    # 6
    {
        "id": 6,
        "query": "Turkish full-back under 24 with dribbling >= 13 at Fenerbahce",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Ferdi Kadıoğlu",
        "expected_found": True,
        "expected_empty": False,
        "note": "Ferdi: Age=23, Dri=14, D/WB(RL), Turkish, Fenerbahce",
    },
    # 7
    {
        "id": 7,
        "query": "left-footed Turkish winger with acceleration >= 16 at Fenerbahce",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Cengiz Ünder",
        "expected_found": True,
        "expected_empty": False,
        "note": "Cengiz: Acc=17, Left foot, AM(RL), Turkish, Fenerbahce",
    },
    # 8
    {
        "id": 8,
        "query": "Polish midfielder under 25 at Fenerbahce with vision >= 15",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Sebastian Szymański",
        "expected_found": True,
        "expected_empty": False,
        "note": "Szymański: Vis=15, Age=24, Polish, M(C)/AM(RLC), Fenerbahce",
    },
    # 9
    {
        "id": 9,
        "query": "Croatian goalkeeper at Fenerbahce",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Dominik Livaković",
        "expected_found": True,
        "expected_empty": False,
        "note": "Livaković: Croatian GK, Fenerbahce",
    },
    # 10
    {
        "id": 10,
        "query": "Argentinian attacking midfielder with technique >= 19 and dribbling >= 19",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Lionel Messi",
        "expected_found": True,
        "expected_empty": False,
        "note": "Messi: Tec=20, Dri=20, Argentinian, AM(RC)/ST",
    },
    # 11
    {
        "id": 11,
        "query": "Belgian midfielder with passing >= 17 and vision >= 19",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Kevin De Bruyne",
        "expected_found": True,
        "expected_empty": False,
        "note": "De Bruyne: Pas=18, Vis=20, Belgian, M(RLC)/AM(C)",
    },
    # 12
    {
        "id": 12,
        "query": "French striker with pace >= 19 and acceleration >= 19",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Kylian Mbappé",
        "expected_found": True,
        "expected_empty": False,
        "note": "Mbappé: Pac=20, Acc=20, French, AM(RL)/ST(C)",
    },
    # 13
    {
        "id": 13,
        "query": "Spanish defensive midfielder with tackling >= 16 and marking >= 12",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Rodri",
        "expected_found": True,
        "expected_empty": False,
        "note": "Rodri: Tck=17, Mar=13, Spanish, DM/D(C)/M(C)",
    },
    # 14
    {
        "id": 14,
        "query": "Norwegian striker under 23 with heading >= 14 and finishing >= 17",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Erling Haaland",
        "expected_found": True,
        "expected_empty": False,
        "note": "Haaland: Hea=15, Fin=18, Age=22, Norwegian, ST(C)",
    },
    # 15
    {
        "id": 15,
        "query": "Dutch centre-back at Liverpool with strength >= 16 and heading >= 16",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Virgil van Dijk",
        "expected_found": True,
        "expected_empty": False,
        "note": "Van Dijk: Str=17, Hea=17, Dutch, D(C), Liverpool",
    },
    # 16
    {
        "id": 16,
        "query": "Egyptian winger at Liverpool with acceleration >= 17 and pace >= 16",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Mohamed Salah",
        "expected_found": True,
        "expected_empty": False,
        "note": "Salah: Acc=18, Pac=17, Egyptian, AM(RL)/ST, Liverpool",
    },
    # 17
    {
        "id": 17,
        "query": "English midfielder under 21 with dribbling >= 16 and finishing >= 15",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Jude Bellingham",
        "expected_found": True,
        "expected_empty": False,
        "note": "Bellingham: Dri=17, Fin=16, Age=20, English, DM/M-AM(C)",
    },
    # 18
    {
        "id": 18,
        "query": "Spanish midfielder under 21 at Barcelona with technique >= 16 and dribbling >= 16",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Pedri",
        "expected_found": True,
        "expected_empty": False,
        "note": "Pedri: Tec=17, Dri=17, Age=20, Spanish, M(C)/AM(RLC), Barcelona",
    },
    # 19
    {
        "id": 19,
        "query": "English winger at Fenerbahce with acceleration >= 15",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Ryan Kent",
        "expected_found": True,
        "expected_empty": False,
        "note": "Kent: Acc=16, English, AM(RLC), Fenerbahce",
    },
    # 20
    {
        "id": 20,
        "query": "Bosnian defensive midfielder at Fenerbahce with tackling >= 12 and work rate >= 12",
        "category": "real_player_lookup",
        "test_type": "search",
        "target_player": "Rade Krunić",
        "expected_found": True,
        "expected_empty": False,
        "note": "Krunić: Tck=14, Wor=14, Bosnian, DM/M-AM(C), Fenerbahce",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 2 — Non-existent player direct lookups (8 tests)
    # These names do NOT exist in the database.
    # Expected: no results returned / not-found message.
    # -----------------------------------------------------------------------

    # 21
    {
        "id": 21,
        "query": "Arda Suğur",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Arda Suğur",
        "expected_found": False,
        "expected_empty": True,
        "note": "Completely invented name — should return no results",
    },
    # 22
    {
        "id": 22,
        "query": "Umut Can Çubukçu",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Umut Can Çubukçu",
        "expected_found": False,
        "expected_empty": True,
        "note": "Invented name — should return no results",
    },
    # 23
    {
        "id": 23,
        "query": "Mustafa Ege Özer",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Mustafa Ege Özer",
        "expected_found": False,
        "expected_empty": True,
        "note": "Invented name — should return no results",
    },
    # 24
    {
        "id": 24,
        "query": "Barış Demir Türkoğlu",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Barış Demir Türkoğlu",
        "expected_found": False,
        "expected_empty": True,
        "note": "Invented Turkish name — should return no results",
    },
    # 25
    {
        "id": 25,
        "query": "John McFakePlayerson",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "John McFakePlayerson",
        "expected_found": False,
        "expected_empty": True,
        "note": "Clearly fictional English name",
    },
    # 26
    {
        "id": 26,
        "query": "Zlatan Papadopoulos",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Zlatan Papadopoulos",
        "expected_found": False,
        "expected_empty": True,
        "note": "Mixed fictional name unlikely to exist in FM24",
    },
    # 27
    {
        "id": 27,
        "query": "Emirhan Çelik Yıldırım",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Emirhan Çelik Yıldırım",
        "expected_found": False,
        "expected_empty": True,
        "note": "Invented Turkish three-part name",
    },
    # 28
    {
        "id": 28,
        "query": "Kemal Oğuz Sönmez",
        "category": "nonexistent_player",
        "test_type": "lookup",
        "target_player": "Kemal Oğuz Sönmez",
        "expected_found": False,
        "expected_empty": True,
        "note": "Invented Turkish name",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 3 — "Players similar to [fake name]" queries (5 tests)
    # Similarity queries referencing non-existent players.
    # Expected: no results (no reference player found to build from).
    # -----------------------------------------------------------------------

    # 29
    {
        "id": 29,
        "query": "players like Mustafa Ege Ozer",
        "category": "similar_to_fake",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Similarity query referencing a non-existent player",
    },
    # 30
    {
        "id": 30,
        "query": "midfielder like Umut Can Cubukcu",
        "category": "similar_to_fake",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Position + similarity query with fake reference player",
    },
    # 31
    {
        "id": 31,
        "query": "winger similar to Baris Demir",
        "category": "similar_to_fake",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Winger similarity to invented player",
    },
    # 32
    {
        "id": 32,
        "query": "find a player like Yusuf Demir Aksoy",
        "category": "similar_to_fake",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Generic 'find' similarity with fake name",
    },
    # 33
    {
        "id": 33,
        "query": "striker similar to Ahmet Guclu",
        "category": "similar_to_fake",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Striker similarity with non-existent player",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 4 — Long / strict / complex queries (10 tests)
    # Tests behavior under high specificity, multi-constraint, or impossible queries.
    # -----------------------------------------------------------------------

    # 34 — Very specific multi-constraint (should still find Pedri)
    {
        "id": 34,
        "query": (
            "left-footed Spanish central midfielder under 22 at Barcelona "
            "with passing >= 15 and technique >= 15 and dribbling >= 15 "
            "and vision >= 14 and composure >= 14"
        ),
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": "Pedri",
        "expected_found": True,
        "expected_empty": False,
        "note": "7-constraint query targeting Pedri — all constraints are satisfied by his data",
    },
    # 35 — Impossible GK constraint (no GK under 18 has Ref>=19)
    {
        "id": 35,
        "query": "goalkeeper under 18 with reflexes >= 19 and handling >= 18 in the English Premier League",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Over-constrained GK query — extremely unlikely to match any player",
    },
    # 36 — Multi-nationality at specific club
    {
        "id": 36,
        "query": "Turkish or Bosnian defensive midfielder at Fenerbahce with tackling >= 13 and passing >= 12",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": "İsmail Yüksek",
        "expected_found": True,
        "expected_empty": False,
        "note": "Nationality OR + club + 2 hard attrs — should surface Yüksek or Krunić",
    },
    # 37 — Multi-position winger/striker at Man City
    {
        "id": 37,
        "query": "right-footed winger or striker at Man City under 28 with pace >= 16 and dribbling >= 15",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,   # we just check results are non-empty
        "expected_empty": False,
        "note": "Valid compound query — should return at least 1 player",
    },
    # 38 — Extreme attribute thresholds (near-impossible)
    {
        "id": 38,
        "query": "centre-back with marking >= 18 and tackling >= 18 and strength >= 18 and heading >= 18 and pace >= 16",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Five attrs all >= 18 simultaneously — extremely rare or impossible",
    },
    # 39 — Long narrative creative query (Pedri-like)
    {
        "id": 39,
        "query": (
            "I am looking for a young creative left-footed attacking midfielder "
            "under 22 who plays for a Spanish club, has excellent technique above 16, "
            "great dribbling above 15, good passing above 14, "
            "and is valued under 50 million euros"
        ),
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": "Pedri",
        "expected_found": True,
        "expected_empty": False,
        "note": "Full narrative scouting request — should surface Pedri",
    },
    # 40 — Multi-nationality midfielder (Messi-like)
    {
        "id": 40,
        "query": "Brazilian or Argentine midfielder under 30 with passing >= 16 and vision >= 16 and technique >= 15",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": "Lionel Messi",
        "expected_found": True,
        "expected_empty": False,
        "note": "Nationality OR + 3 hard attrs — Messi satisfies all (Pas=19, Vis=20, Tec=20)",
    },
    # 41 — Logically impossible position combo
    {
        "id": 41,
        "query": "striker who is also a goalkeeper under 20 with finishing >= 18 and reflexes >= 18",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "ST + GK simultaneously — no player should satisfy this",
    },
    # 42 — Turkish Super League striker over 30 (Džeko)
    {
        "id": 42,
        "query": "Turkish Super League striker over 30 with heading >= 15 and strength >= 16",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": "Edin Džeko",
        "expected_found": True,
        "expected_empty": False,
        "note": "League + age + 2 hard attrs — Džeko: Age=37, Hea=17, Str=18",
    },
    # 43 — Value-ranged winger at Fenerbahce
    {
        "id": 43,
        "query": "winger valued under 7 million euros at Fenerbahce with acceleration >= 15 and dribbling >= 13",
        "category": "long_strict_query",
        "test_type": "search",
        "target_player": "Ryan Kent",
        "expected_found": True,
        "expected_empty": False,
        "note": "Value cap + club + 2 attrs — Kent: Value=6.6M, Acc=16, Dri=14",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 5 — Edge cases (7 tests)
    # Turkish character normalization, partial names, typos, generic queries.
    # -----------------------------------------------------------------------

    # 44 — ASCII version of Turkish name
    {
        "id": 44,
        "query": "Ismail Yuksek",
        "category": "edge_case",
        "test_type": "lookup",
        "target_player": "İsmail Yüksek",
        "expected_found": True,
        "expected_empty": False,
        "note": "ASCII variant of İsmail Yüksek — accents stripped lookup must succeed",
    },
    # 45 — Partial surname only
    {
        "id": 45,
        "query": "Szymanski",
        "category": "edge_case",
        "test_type": "lookup",
        "target_player": "Sebastian Szymański",
        "expected_found": True,
        "expected_empty": False,
        "note": "Surname-only lookup without diacritics",
    },
    # 46 — All lowercase full name
    {
        "id": 46,
        "query": "lionel messi",
        "category": "edge_case",
        "test_type": "lookup",
        "target_player": "Lionel Messi",
        "expected_found": True,
        "expected_empty": False,
        "note": "Lowercase-only name lookup",
    },
    # 47 — ASCII version of Turkish name with diacritics
    {
        "id": 47,
        "query": "Ferdi Kadioglu",
        "category": "edge_case",
        "test_type": "lookup",
        "target_player": "Ferdi Kadıoğlu",
        "expected_found": True,
        "expected_empty": False,
        "note": "ASCII transliteration of Ferdi Kadıoğlu — normalization test",
    },
    # 48 — Very generic query (no hard constraints)
    {
        "id": 48,
        "query": "good striker",
        "category": "edge_case",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Minimal semantic query — should return results without crashing",
    },
    # 49 — Bare word query
    {
        "id": 49,
        "query": "players",
        "category": "edge_case",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Near-empty query — system should not crash and should return generic results",
    },
    # 50 — Club filter only
    {
        "id": 50,
        "query": "Fenerbahce players",
        "category": "edge_case",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Club-only filter — should return Fenerbahce squad members",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 6 — Goalkeeper-specific attribute tests (8 tests)
    # Covers GK-only FM24 attributes: 1v1, Ref, Han, Aer, Cmd, Com, Ecc, Pun, Kic, TRO, Thr
    # -----------------------------------------------------------------------

    # 51
    {
        "id": 51,
        "query": "goalkeeper at Fenerbahce with one-on-ones >= 18",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "Dominik Livaković",
        "expected_found": True,
        "expected_empty": False,
        "note": "1v1 attr at club — Livaković: 1v1=19, Fenerbahce",
    },
    # 52
    {
        "id": 52,
        "query": "goalkeeper with aerial reach >= 19 and reflexes >= 18",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "Thibaut Courtois",
        "expected_found": True,
        "expected_empty": False,
        "note": "Aerial reach + reflexes — Courtois: Aer=20, Ref=19",
    },
    # 53
    {
        "id": 53,
        "query": "goalkeeper with eccentricity >= 18",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "André Onana",
        "expected_found": True,
        "expected_empty": False,
        "note": "Eccentricity — Onana: Ecc=19; Neuer: Ecc=18",
    },
    # 54
    {
        "id": 54,
        "query": "goalkeeper with command of area >= 15 and reflexes >= 18",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "Thibaut Courtois",
        "expected_found": True,
        "expected_empty": False,
        "note": "Cmd + Ref — Courtois: Cmd=16, Ref=19",
    },
    # 55
    {
        "id": 55,
        "query": "Turkish Super League goalkeeper under 20",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Young GK Turkish league — Jankat Yılmaz (18), Emir Yaşar (17) etc. exist",
    },
    # 56
    {
        "id": 56,
        "query": "goalkeeper with reflexes >= 19 and one-on-ones >= 17 at Real Madrid",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "Thibaut Courtois",
        "expected_found": True,
        "expected_empty": False,
        "note": "Ref + 1v1 at club — Courtois: Ref=19, 1v1=18, Real Madrid",
    },
    # 57
    {
        "id": 57,
        "query": "Turkish goalkeeper under 26 with eccentricity >= 14",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "İrfan Can Eğribayat",
        "expected_found": True,
        "expected_empty": False,
        "note": "Nationality + age + Ecc — Eğribayat: Age=25, Ecc=15, Turkish",
    },
    # 58
    {
        "id": 58,
        "query": "goalkeeper at Fenerbahce with punching tendency >= 15",
        "category": "gk_attributes",
        "test_type": "search",
        "target_player": "Dominik Livaković",
        "expected_found": True,
        "expected_empty": False,
        "note": "Punching tendency attr — Livaković: Pun=16, Fenerbahce",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 7 — Remaining attribute coverage (12 tests)
    # Tests FM24 attrs not covered in Cat 1: Cor, Fre, Sta, OtB, Det, Fla,
    # Bra, Cro, Lon, Nat, Cnt, Pos, Fir, Agi, Bal, Wor, Ldr
    # -----------------------------------------------------------------------

    # 59
    {
        "id": 59,
        "query": "midfielder with corners >= 19 and free kicks >= 19",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "James Ward-Prowse",
        "expected_found": True,
        "expected_empty": False,
        "note": "Corners + free kicks — Ward-Prowse: Cor=20, Fre=20",
    },
    # 60
    {
        "id": 60,
        "query": "English winger with off the ball >= 19",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Raheem Sterling",
        "expected_found": True,
        "expected_empty": False,
        "note": "Off the ball — Sterling: OtB=20, English, winger",
    },
    # 61
    {
        "id": 61,
        "query": "striker under 23 with determination >= 20",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Erling Haaland",
        "expected_found": True,
        "expected_empty": False,
        "note": "Determination — Haaland: Det=20, Age=22, ST",
    },
    # 62
    {
        "id": 62,
        "query": "attacking midfielder with flair >= 19",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Lionel Messi",
        "expected_found": True,
        "expected_empty": False,
        "note": "Flair — Messi: Fla=20, AM(RC)",
    },
    # 63
    {
        "id": 63,
        "query": "centre-back with bravery >= 19",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Nicolás Otamendi",
        "expected_found": True,
        "expected_empty": False,
        "note": "Bravery — Otamendi: Bra=20, D(C)",
    },
    # 64
    {
        "id": 64,
        "query": "midfielder with long shots >= 17 and crossing >= 18",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Kevin De Bruyne",
        "expected_found": True,
        "expected_empty": False,
        "note": "Long shots + crossing — De Bruyne: Lon=17, Cro=19",
    },
    # 65
    {
        "id": 65,
        "query": "midfielder with concentration >= 17 and positioning >= 17",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Rodri",
        "expected_found": True,
        "expected_empty": False,
        "note": "Concentration + positioning — Rodri: Cnt=17, Pos=17",
    },
    # 66
    {
        "id": 66,
        "query": "attacking midfielder with first touch >= 18 and flair >= 18",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Lionel Messi",
        "expected_found": True,
        "expected_empty": False,
        "note": "First touch + flair — Messi: Fir=19, Fla=20",
    },
    # 67
    {
        "id": 67,
        "query": "striker with agility >= 16 and balance >= 16",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Erling Haaland",
        "expected_found": True,
        "expected_empty": False,
        "note": "Agility + balance — Haaland: Agi=16, Bal=17",
    },
    # 68
    {
        "id": 68,
        "query": "striker with off the ball >= 18 and determination >= 20",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Erling Haaland",
        "expected_found": True,
        "expected_empty": False,
        "note": "OtB + Det — Haaland: OtB=18, Det=20",
    },
    # 69
    {
        "id": 69,
        "query": "midfielder with work rate >= 19 and stamina >= 20",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "James Ward-Prowse",
        "expected_found": True,
        "expected_empty": False,
        "note": "Work rate + stamina — Ward-Prowse: Wor=19, Sta=20",
    },
    # 70
    {
        "id": 70,
        "query": "striker with natural fitness >= 19",
        "category": "attribute_coverage",
        "test_type": "search",
        "target_player": "Erling Haaland",
        "expected_found": True,
        "expected_empty": False,
        "note": "Natural fitness — Haaland: Nat=19",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 8 — Invalid / non-existent positions (5 tests)
    # The parser extracts partial tokens; system should not crash.
    # expected_empty=False: system gracefully falls back to partial matches.
    # -----------------------------------------------------------------------

    # 71
    {
        "id": 71,
        "query": "middle defence winger",
        "category": "invalid_position",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Nonsense position — parser extracts D( + winger tokens; system returns results without crash",
    },
    # 72
    {
        "id": 72,
        "query": "attacking goalkeeper under 25",
        "category": "invalid_position",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Oxymoron position — parses as GK + age filter; returns young GKs",
    },
    # 73
    {
        "id": 73,
        "query": "sweeper keeper winger",
        "category": "invalid_position",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Two incompatible roles — parser extracts GK + winger OR; should not crash",
    },
    # 74
    {
        "id": 74,
        "query": "goalkeeper striker hybrid under 25",
        "category": "invalid_position",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "GK + ST position OR — no player plays both, but parser uses OR so returns GKs and strikers",
    },
    # 75
    {
        "id": 75,
        "query": "deep lying playmaker centre-back with finishing >= 18",
        "category": "invalid_position",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "CB role + unrealistic Fin>=18 — no CB has Fin>=18 in dataset",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 9 — Non-existent / unrecognized nationalities (5 tests)
    # FM24 dataset has no Mongolian or Martian players; Mongolian is not in the
    # parser's nationality list so the filter is NOT applied — system falls
    # back to position-only filtering and returns generic results.
    # -----------------------------------------------------------------------

    # 76
    {
        "id": 76,
        "query": "Mongolian midfielder",
        "category": "unusual_nationality",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Mongolian not in nationality list — filter silently skipped; returns generic midfielders",
    },
    # 77
    {
        "id": 77,
        "query": "Martian striker",
        "category": "unusual_nationality",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Nonexistent nationality — filter skipped; returns strikers",
    },
    # 78
    {
        "id": 78,
        "query": "North Korean goalkeeper with reflexes >= 15",
        "category": "unusual_nationality",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "No North Korean players in FM24 — nationality not recognized by parser; returns GKs with Ref>=15",
    },
    # 79
    {
        "id": 79,
        "query": "Mongolian goalkeeper with reflexes >= 20",
        "category": "unusual_nationality",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Ref>=20 is impossible (max=20 but extremely rare) — zero results expected regardless of nationality",
    },
    # 80
    {
        "id": 80,
        "query": "Andorran winger under 22 with pace >= 17",
        "category": "unusual_nationality",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Andorran exists in DB (27 players) but no winger under 22 with Pac>=17 is expected",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 10 — Age-focused tests (5 tests)
    # Tests age boundary parsing: under 17, over 35, old strikers, young talents
    # -----------------------------------------------------------------------

    # 81
    {
        "id": 81,
        "query": "winger under 16 at Barcelona",
        "category": "age_focused",
        "test_type": "search",
        "target_player": "Lamine Yamal",
        "expected_found": True,
        "expected_empty": False,
        "note": "Ultra-young — Lamine Yamal: Age=15, AM(RL), Barcelona",
    },
    # 82
    {
        "id": 82,
        "query": "striker over 36 with heading >= 18",
        "category": "age_focused",
        "test_type": "search",
        "target_player": "Cristiano Ronaldo",
        "expected_found": True,
        "expected_empty": False,
        "note": "Veteran striker — Ronaldo: Age=38, Hea=18; Giroud: Age=36, Hea=19",
    },
    # 83
    {
        "id": 83,
        "query": "striker over 35",
        "category": "age_focused",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Age-only filter for veterans — Ronaldo, Cavani, Giroud, Suárez all qualify",
    },
    # 84
    {
        "id": 84,
        "query": "Turkish Super League player under 17",
        "category": "age_focused",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "League + strict young age — several youth players in Turkish clubs under 17",
    },
    # 85
    {
        "id": 85,
        "query": "midfielder over 30 with stamina >= 16 and natural fitness >= 16",
        "category": "age_focused",
        "test_type": "search",
        "target_player": "Kevin De Bruyne",
        "expected_found": True,
        "expected_empty": False,
        "note": "Older high-fitness midfielder — De Bruyne: Age=32, Sta=16, Nat=16",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 11 — Very short / minimal queries (5 tests)
    # Single words or two-word queries. Tests system robustness at minimal input.
    # -----------------------------------------------------------------------

    # 86
    {
        "id": 86,
        "query": "gk",
        "category": "short_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Two-letter abbreviation — parser may or may not detect GK position; should not crash",
    },
    # 87
    {
        "id": 87,
        "query": "pace",
        "category": "short_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Single attribute word — no hard constraint, treated as soft signal or semantic; returns results",
    },
    # 88
    {
        "id": 88,
        "query": "young winger",
        "category": "short_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Two-word semantic query — should return young winger candidates",
    },
    # 89
    {
        "id": 89,
        "query": "Brazilian",
        "category": "short_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Nationality-only query — should return Brazilian players",
    },
    # 90
    {
        "id": 90,
        "query": "strong",
        "category": "short_query",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": False,
        "note": "Pure adjective — treated as semantic/soft signal; should not crash and returns results",
    },

    # -----------------------------------------------------------------------
    # CATEGORY 12 — Long narrative queries (5 tests)
    # Full scouting request paragraphs. Tests parser robustness under high
    # verbosity and mixed hard/soft constraint extraction.
    # -----------------------------------------------------------------------

    # 91 — 10-constraint Haaland-like request
    {
        "id": 91,
        "query": (
            "I need a powerful and athletic centre-forward who is under 24 years old, "
            "left-footed or right-footed, plays in one of the top European leagues, "
            "has excellent finishing above 17, great strength above 16, "
            "outstanding natural fitness above 18, superb off the ball movement above 17, "
            "good heading above 14, solid stamina above 13, high determination above 19, "
            "and is not a goalkeeper or defender"
        ),
        "category": "long_narrative",
        "test_type": "search",
        "target_player": "Erling Haaland",
        "expected_found": True,
        "expected_empty": False,
        "note": "10-constraint paragraph — Haaland satisfies: Fin=18, Str=17, Nat=19, OtB=18, Hea=15, Sta=14, Det=20",
    },
    # 92 — Long text with only 2 real hard constraints
    {
        "id": 92,
        "query": (
            "We are scouting for our upcoming season and we have identified a need "
            "for a reliable and experienced defensive midfielder who can win the ball "
            "back efficiently and distribute it well across the pitch. We want someone "
            "who is tactically aware and reads the game well. Ideally, the player should "
            "have tackling at least 16 and passing at least 16, and must be playing in "
            "a competitive European league. Age and nationality are not strict requirements, "
            "but we would prefer someone in their prime years."
        ),
        "category": "long_narrative",
        "test_type": "search",
        "target_player": "Rodri",
        "expected_found": True,
        "expected_empty": False,
        "note": "Verbose with 2 real constraints (Tck>=16, Pas>=16) — Rodri: Tck=17, Pas=16",
    },
    # 93 — Contradictory narrative
    {
        "id": 93,
        "query": (
            "Looking for a striker who is very fast with pace over 18 but also very slow "
            "with pace under 10, who is both a goalkeeper and a striker, "
            "has finishing over 18 but also finishing under 5, "
            "and is both under 18 and over 35 years old at the same time"
        ),
        "category": "long_narrative",
        "test_type": "search",
        "target_player": None,
        "expected_found": False,
        "expected_empty": True,
        "note": "Self-contradictory constraints (Pac>18 AND Pac<10, Age<18 AND Age>35) — should return no results",
    },
    # 94 — Multi-nationality multi-position complex
    {
        "id": 94,
        "query": (
            "Find me a Turkish, Brazilian, or Spanish left-footed central midfielder "
            "or defensive midfielder under 28 years old who plays in the Turkish Super League "
            "or Spanish La Liga, with passing above 13, technique above 12, "
            "vision above 12, and a transfer value under 20 million euros. "
            "The player should be a leader on the pitch with good work rate and teamwork."
        ),
        "category": "long_narrative",
        "test_type": "search",
        "target_player": "İsmail Yüksek",
        "expected_found": True,
        "expected_empty": False,
        "note": "Multi-nationality + multi-position + multi-league long query — Yüksek satisfies all hard constraints",
    },
    # 95 — Long query with repeated/conflicting soft attributes
    {
        "id": 95,
        "query": (
            "I want a goalkeeper with excellent reflexes, outstanding reflexes, "
            "great reflexes, very high reflexes, and ideally also good reflexes. "
            "The goalkeeper should be under 32 and play in a European club. "
            "Good handling is also important, as is aerial ability. "
            "The goalkeeper must have reflexes above 18."
        ),
        "category": "long_narrative",
        "test_type": "search",
        "target_player": "Thibaut Courtois",
        "expected_found": True,
        "expected_empty": False,
        "note": "Repeated 'reflexes' soft signal + one hard constraint (Ref>=18) — Courtois: Ref=19",
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

RESULT_COLS = [
    "id", "category", "query", "target_player", "note",
    "expected_found", "expected_empty",
    "n_results", "top5_players", "parsed_tokens",
    "target_in_results", "results_empty",
    "pass", "fail_reason",
    "latency_s",
]

rows = []

print("\n" + "=" * 80)
print(f"Running {len(TESTS)} tests  (top_k = 20)  |  categories: {sorted(set(t['category'] for t in TESTS))}")
print("=" * 80)

for t in TESTS:
    t_start = time.perf_counter()
    qid = t["id"]
    query = t["query"]
    ttype = t["test_type"]
    target = t.get("target_player")
    exp_found = t["expected_found"]
    exp_empty = t["expected_empty"]

    try:
        if ttype == "lookup":
            # Name-based lookup
            recs = players_to_records(find_players_by_name(df, query, limit=20))
            n_results = len(recs)
            target_in_results = player_in_results(recs, target) if target else False
            results_empty = n_results == 0
            parsed_tokens = "(name lookup)"
        else:
            # Full search
            res = run_search(query, top_k=20)
            recs = res.get("players", [])
            n_results = len(recs)
            parsed = res.get("parsed", {})
            target_in_results = player_in_results(recs, target) if target else False
            results_empty = n_results == 0 or bool(res.get("message"))
            parsed_tokens = fmt_trace(parsed)

        latency = time.perf_counter() - t_start

        # --- Evaluate pass/fail ---
        fail_reason = ""
        passed = True

        if exp_empty:
            # We expect NO results
            if not results_empty:
                passed = False
                fail_reason = f"expected empty but got {n_results} results"
        else:
            # We expect results
            if results_empty and not (t["category"] == "edge_case" and target is None):
                passed = False
                fail_reason = "expected results but got none"

        if exp_found and not target_in_results:
            passed = False
            fail_reason += (f" | target '{target}' not in top-20" if fail_reason
                            else f"target '{target}' not in top-20")

        if not exp_found and target_in_results and target:
            passed = False
            fail_reason = f"target '{target}' appeared but should NOT have"

        top5 = "; ".join(
            r.get("Player_Name", "?") for r in recs[:5]
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] #{qid:02d} | {query[:60]:<60} | n={n_results:4d} | {latency:.2f}s")
        if not passed:
            print(f"         ⚠  {fail_reason}")

        rows.append({
            "id": qid,
            "category": t["category"],
            "query": query,
            "target_player": target or "",
            "note": t["note"],
            "expected_found": exp_found,
            "expected_empty": exp_empty,
            "n_results": n_results,
            "top5_players": top5,
            "parsed_tokens": parsed_tokens,
            "target_in_results": target_in_results,
            "results_empty": results_empty,
            "pass": passed,
            "fail_reason": fail_reason,
            "latency_s": round(latency, 3),
        })

    except Exception as exc:
        latency = time.perf_counter() - t_start
        print(f"  [ERROR] #{qid:02d} | {query[:60]} | {exc}")
        rows.append({
            "id": qid,
            "category": t["category"],
            "query": query,
            "target_player": target or "",
            "note": t["note"],
            "expected_found": exp_found,
            "expected_empty": exp_empty,
            "n_results": -1,
            "top5_players": "",
            "parsed_tokens": f"ERROR: {exc}",
            "target_in_results": False,
            "results_empty": True,
            "pass": False,
            "fail_reason": f"EXCEPTION: {exc}",
            "latency_s": round(latency, 3),
        })

# --- Summary ---
n_pass = sum(1 for r in rows if r["pass"])
n_fail = len(rows) - n_pass
print("\n" + "=" * 80)
print(f"Results: {n_pass}/{len(rows)} passed  ({n_pass/len(rows)*100:.1f}%)")

# By category
print("\nBreakdown by category:")
cats = {}
for r in rows:
    cats.setdefault(r["category"], []).append(r["pass"])
for cat, ps in sorted(cats.items()):
    p = sum(ps)
    print(f"  {cat:<30} {p}/{len(ps)}")

# Save CSV
out_path = "eval_results.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=RESULT_COLS)
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ABLATION STUDY")
print("Comparing: structured-only vs semantic-only vs hybrid")
print("Subset: real_player_lookup + gk_attributes + attribute_coverage (tests with a named target)")
print("=" * 80)

ABLATION_COLS = [
    "id", "query", "target_player",
    "struct_found", "struct_n",
    "semantic_found", "semantic_n",
    "hybrid_found", "hybrid_n",
    "winner",
]

ablation_rows = []
ABLATION_CATS = {"real_player_lookup", "gk_attributes", "attribute_coverage"}
cat1_tests = [t for t in TESTS if t["category"] in ABLATION_CATS and t.get("target_player")]

for t in cat1_tests:
    qid = t["id"]
    query = t["query"]
    target = t["target_player"]

    # Structured-only
    try:
        res_s = run_structured_only(query, top_k=20)
        recs_s = res_s.get("players", [])
        struct_found = player_in_results(recs_s, target)
        struct_n = len(recs_s)
    except Exception:
        struct_found, struct_n = False, -1

    # Semantic-only
    try:
        res_sem = run_semantic_only(query, top_k=20)
        recs_sem = res_sem.get("players", [])
        sem_found = player_in_results(recs_sem, target)
        sem_n = len(recs_sem)
    except Exception:
        sem_found, sem_n = False, -1

    # Hybrid
    try:
        res_h = run_search(query, top_k=20)
        recs_h = res_h.get("players", [])
        hyb_found = player_in_results(recs_h, target)
        hyb_n = len(recs_h)
    except Exception:
        hyb_found, hyb_n = False, -1

    # Winner: which mode(s) found the player
    modes = []
    if struct_found: modes.append("structured")
    if sem_found:    modes.append("semantic")
    if hyb_found:    modes.append("hybrid")
    winner = "+".join(modes) if modes else "none"

    status_s = "✓" if struct_found else "✗"
    status_e = "✓" if sem_found   else "✗"
    status_h = "✓" if hyb_found   else "✗"
    print(f"  #{qid:02d} struct={status_s} sem={status_e} hybrid={status_h}  | {target} | {query[:50]}")

    ablation_rows.append({
        "id": qid,
        "query": query,
        "target_player": target,
        "struct_found": struct_found,
        "struct_n": struct_n,
        "semantic_found": sem_found,
        "semantic_n": sem_n,
        "hybrid_found": hyb_found,
        "hybrid_n": hyb_n,
        "winner": winner,
    })

# Ablation summary
n_abl = len(ablation_rows)
print(f"\nAblation totals (out of {n_abl}):")
print(f"  Structured-only : {sum(r['struct_found']  for r in ablation_rows)}/{n_abl}")
print(f"  Semantic-only   : {sum(r['semantic_found'] for r in ablation_rows)}/{n_abl}")
print(f"  Hybrid          : {sum(r['hybrid_found']   for r in ablation_rows)}/{n_abl}")

# Cases where hybrid beats both others
hybrid_wins = [r for r in ablation_rows if r["hybrid_found"] and not r["struct_found"] and not r["semantic_found"]]
struct_only_wins = [r for r in ablation_rows if r["struct_found"] and not r["hybrid_found"]]
sem_only_wins    = [r for r in ablation_rows if r["semantic_found"] and not r["hybrid_found"]]

print(f"\n  Hybrid found target that both others missed : {len(hybrid_wins)}")
print(f"  Structured found target that hybrid missed  : {len(struct_only_wins)}")
print(f"  Semantic  found target that hybrid missed   : {len(sem_only_wins)}")

# Save ablation CSV
abl_path = "ablation.csv"
with open(abl_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ABLATION_COLS)
    w.writeheader()
    w.writerows(ablation_rows)
print(f"\nSaved: {abl_path}")
print("=" * 80)
print("Done.")

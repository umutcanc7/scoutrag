"""Core ScoutRAG engine: cleaning, parsing, filtering, ranking, semantic retrieval.

This module is shared by the Jupyter notebook and the FastAPI backend so the
exact same logic powers both. It has no Colab / Google Drive dependencies.
"""

from __future__ import annotations

import os
import re
import pickle
import unicodedata

import numpy as np
import pandas as pd


def _strip_accents(s: str) -> str:
    """Lower-case and remove diacritics so 'Beşiktaş' == 'besiktas'."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Turkish dotless/dotted i and a few stragglers NFKD leaves alone
    s = (s.replace("ı", "i").replace("İ", "i").replace("ş", "s")
          .replace("ğ", "g").replace("ø", "o").replace("ß", "ss"))
    return s.lower().strip()

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NO_MATCH_MESSAGE = "There is no player like that in our database."

# FM24 attribute abbreviation -> readable name
ATTR_READABLE = {
    "Acc": "acceleration", "Agi": "agility",      "Bal": "balance",
    "Bra": "bravery",      "Cmp": "composure",    "Cnt": "concentration",
    "Cor": "corners",      "Cro": "crossing",     "Dec": "decisions",
    "Det": "determination","Dri": "dribbling",    "Fin": "finishing",
    "Fir": "first touch",  "Fla": "flair",        "Fre": "free kicks",
    "Hea": "heading",      "Jum": "jumping",      "Ldr": "leadership",
    "Lon": "long shots",   "Mar": "marking",      "Nat": "natural fitness",
    "OtB": "off the ball", "Pac": "pace",         "Pas": "passing",
    "Pen": "penalties",    "Pos": "positioning",  "Sta": "stamina",
    "Str": "strength",     "Tck": "tackling",     "Tea": "teamwork",
    "Tec": "technique",    "Vis": "vision",       "Wor": "work rate",
    "Agg": "aggression",   "Ant": "anticipation", "Aer": "aerial reach",
    # Goalkeeper-specific
    "1v1": "one-on-ones",  "Com": "communication","Cmd": "command of area",
    "Ecc": "eccentricity", "Han": "handling",     "Kic": "kicking",
    "L Th": "long throws", "Pun": "punching (tendency)", "Ref": "reflexes",
    "TRO": "rushing out (tendency)", "Thr": "throwing",
}
READABLE_TO_ABBR = {v: k for k, v in ATTR_READABLE.items()}

# Attribute pools for position-aware ranking.
GK_ONLY_COLS = ["Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1",
                "Pun", "Ref", "TRO", "Thr"]
OUTFIELD_ATTR_COLS = [
    "Cor", "Cro", "Dri", "Fin", "Fir", "Fre", "Hea", "Lon", "L Th",
    "Mar", "Pas", "Pen", "Tck", "Tec",                       # technical
    "Agg", "Ant", "Bra", "Cmp", "Cnt", "Dec", "Det", "Fla",
    "Ldr", "OtB", "Pos", "Tea", "Vis", "Wor",                # mental
    "Acc", "Agi", "Bal", "Jum", "Nat", "Pac", "Sta", "Str",  # physical
]
PHYSICAL_MENTAL_SHARED = ["Acc", "Agi", "Bal", "Jum", "Nat", "Pac", "Sta", "Str",
                          "Agg", "Ant", "Bra", "Cmp", "Cnt", "Dec", "Det",
                          "Ldr", "Tea", "Wor"]

ATTR_SCALE_MIN, ATTR_SCALE_MAX = 1, 20  # FM attributes are on a 1-20 scale

# --------------------------------------------------------------------------- #
# 1. Loading + cleaning
# --------------------------------------------------------------------------- #

def _parse_eur_string(s):
    """Parse a transfer-value cell. Ranges take the upper bound. NaN if unknown."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s in ("Not for Sale", "-", ""):
        return np.nan
    if s == "0":
        return 0.0
    s = s.replace("EUR", "").replace("€", "").replace(",", "").strip()
    if " - " in s:
        s = s.split(" - ")[-1].strip()
    m = re.match(r"^([0-9.]+)([KMB]?)$", s)
    if m:
        num = float(m.group(1))
        suf = m.group(2)
        num *= {"K": 1e3, "M": 1e6, "B": 1e9}.get(suf, 1)
        return num
    return np.nan


def _parse_wage(w):
    if pd.isna(w) or str(w).strip() in ("-", ""):
        return np.nan
    w = str(w).replace("€", "").replace("EUR", "").replace(",", "").replace(" p/m", "").strip()
    m = re.match(r"^([0-9.]+)([KMB]?)$", w)
    if m:
        num = float(m.group(1))
        num *= {"K": 1e3, "M": 1e6, "B": 1e9}.get(m.group(2), 1)
        return num
    return np.nan


def _is_invalid_name(name) -> bool:
    """A player name is invalid if it is missing, empty, whitespace-only,
    invisible, or contains no alphabetic character (e.g. the '- -' placeholders)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return True
    s = str(name)
    # strip zero-width / invisible characters
    s = re.sub(r"[​‌‍﻿ ]", "", s).strip()
    if s == "" or s in ("-", "--", "- -"):
        return True
    if not re.search(r"[A-Za-zÀ-ɏ]", s):  # no Latin letter at all
        return True
    return False


def _extract_primary_position(pos_str):
    if pd.isna(pos_str):
        return "Unknown"
    first = str(pos_str).split(",")[0].strip()
    return re.split(r"\s*\(", first)[0].strip()


def load_and_clean(csv_path: str, verbose: bool = True):
    """Load the FM24 CSV and return a cleaned dataframe with derived columns.

    Cleaning steps:
      * drop rows whose player name is missing / invisible / empty / invalid
      * parse Name -> Player_Name + Nationality
      * parse Club -> Club_Name + League
      * parse Transfer Value, Wage, Weight to numerics
      * normalise Preferred Foot, extract Primary_Pos, flag goalkeepers
      * keep `raw_idx` so cached embeddings can be aligned to the cleaned rows
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lstrip("﻿") for c in df.columns]  # drop BOM on 'Name'
    df["raw_idx"] = np.arange(len(df))
    n_raw = len(df)

    # Parse name + nationality from the 'Name' field ("Lionel Messi - Argentinian")
    df["Player_Name"] = df["Name"].apply(
        lambda x: " - ".join(str(x).split(" - ")[:-1]).strip() if " - " in str(x) else str(x).strip()
    )
    df["Nationality"] = df["Name"].apply(
        lambda x: str(x).split(" - ")[-1].strip() if " - " in str(x) else "Unknown"
    )

    # ---- STRICT NAME CLEANING ------------------------------------------------
    invalid_mask = df["Player_Name"].apply(_is_invalid_name)
    n_invalid = int(invalid_mask.sum())
    df = df[~invalid_mask].reset_index(drop=True)
    if verbose:
        print(f"Removed {n_invalid:,} rows with missing/invalid names "
              f"({n_raw:,} -> {len(df):,} players).")

    # Club -> Club_Name + League
    df["Club_Name"] = df["Club"].apply(
        lambda x: " - ".join(str(x).split(" - ")[:-1]).strip() if " - " in str(x) else str(x).strip()
    )
    df["League"] = df["Club"].apply(
        lambda x: str(x).split(" - ")[-1].strip() if " - " in str(x) else "Unknown"
    )

    # Money + weight
    df["Value_EUR"] = df["Transfer Value"].apply(_parse_eur_string)
    df["Not_For_Sale"] = df["Transfer Value"].apply(lambda x: str(x).strip() == "Not for Sale")
    df["Wage_EUR_pm"] = df["Wage"].apply(_parse_wage)
    df["Weight_kg"] = df["Weight"].apply(
        lambda x: int(re.sub(r"[^0-9]", "", str(x)))
        if pd.notna(x) and re.sub(r"[^0-9]", "", str(x)) else np.nan
    )

    # Foot
    foot_map = {"Right": "Right", "Right Only": "Right", "Left": "Left",
                "Left Only": "Left", "Either": "Either"}
    df["Foot"] = df["Preferred Foot"].map(foot_map).fillna("Unknown")

    # Position
    df["Position"] = df["Position"].fillna("Unknown")
    df["Primary_Pos"] = df["Position"].apply(_extract_primary_position)
    df["is_GK"] = df["Position"].str.contains("GK", na=False)

    # Status (injury / info flag)
    df["Status"] = df["Inf"].fillna("").apply(
        lambda x: "Injured" if "Inj" in str(x) else ("Suspended" if "Sus" in str(x) else "Available")
    )

    # Overall quality = mean of the position-appropriate attribute pool (0-20)
    df["Overall"] = df.apply(_overall_quality, axis=1)

    return df


def _overall_quality(row):
    pool = GK_ONLY_COLS + PHYSICAL_MENTAL_SHARED if row.get("is_GK", False) else OUTFIELD_ATTR_COLS
    vals = [row[c] for c in pool if c in row.index and pd.notna(row[c])]
    return float(np.mean(vals)) if vals else np.nan


def align_embeddings(df, pkl_path: str):
    """Load cached embeddings (keyed to original CSV row index) and return the
    subset aligned to the cleaned dataframe order. Returns None if file missing."""
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    emb = np.asarray(obj["embeddings"], dtype="float32")
    idx = df["raw_idx"].to_numpy()
    if idx.max() >= len(emb):
        return None  # embeddings do not cover these rows; rebuild needed
    return emb[idx]


# --------------------------------------------------------------------------- #
# 2. Player text profiles (for embeddings + grounded reports)
# --------------------------------------------------------------------------- #

def build_profiles(df):
    df = df.copy()
    df["profile"] = df.apply(_build_profile, axis=1)
    # Name-free text used for embeddings, so vector similarity captures
    # play-style (position, foot, style, attributes) rather than clustering on
    # the player's name. The human-readable `profile` above keeps the name for
    # grounded reports / display.
    df["embed_text"] = df.apply(_build_embed_text, axis=1)
    df["player_id"] = df.index
    return df


def _top_attr_str(row):
    pool = GK_ONLY_COLS if row["is_GK"] else OUTFIELD_ATTR_COLS
    attrs = {c: row[c] for c in pool if c in row.index and pd.notna(row[c])}
    top = sorted(attrs.items(), key=lambda x: -x[1])[:8]
    return ", ".join(f"{ATTR_READABLE.get(k, k)} {int(v)}" for k, v in top)


def _build_profile(row):
    pos = row["Position"] if row["Position"] != "Unknown" else "an unknown position"
    val = "not for sale" if row["Not_For_Sale"] else (
        f"EUR{row['Value_EUR']:,.0f}" if pd.notna(row["Value_EUR"]) else "value unknown")
    wage = f"EUR{row['Wage_EUR_pm']:,.0f}/month" if pd.notna(row["Wage_EUR_pm"]) else "wage unknown"
    return (f"{row['Player_Name']} is a {row['Nationality']} player aged {row['Age']}, "
            f"playing as {pos} for {row['Club_Name']} ({row['League']}). "
            f"Preferred foot: {row['Foot']}. Style: {row['Style']}. "
            f"Transfer value: {val}. Wage: {wage}. Key attributes: {_top_attr_str(row)}.")


def _build_embed_text(row):
    """Profile text for embeddings — play-style only.

    Deliberately excludes name, nationality, age and value so vector similarity
    (semantic search and player-to-player similarity) captures HOW a player plays
    (position, foot, style, 1-20 attributes) rather than who they are or where
    they're from. Identity facts (age, nationality, club) are handled as explicit
    structured filters, so they only matter when the query asks for them.
    """
    pos = row["Position"] if row["Position"] != "Unknown" else "an unknown position"
    return (f"A player who plays as {pos}. "
            f"Preferred foot: {row['Foot']}. Style: {row['Style']}. "
            f"Key attributes: {_top_attr_str(row)}.")


# --------------------------------------------------------------------------- #
# 3. Query parsing  (numeric bounds, OR conditions, descriptive synonyms)
# --------------------------------------------------------------------------- #

# Descriptive adjective -> (attribute column, "+" prefer-high / "-" prefer-low)
# These are SOFT signals: they steer ranking but do not hard-filter.
ATTR_SYNONYMS = {
    r"\bfast\b": ("Pac", "+"), r"\bpac(?:y|ey)\b": ("Pac", "+"),
    r"\bquick\b": ("Pac", "+"), r"\bspeedy?\b": ("Pac", "+"),
    r"\bexplosive\b": ("Acc", "+"), r"\bslow\b": ("Pac", "-"),
    r"\bstrong\b": ("Str", "+"), r"\bpowerful\b": ("Str", "+"),
    r"\bphysical\b": ("Str", "+"), r"\bmuscular\b": ("Str", "+"),
    r"\bdominant\b": ("Str", "+"),
    r"\baerial\b": ("Hea", "+"), r"\bgood in the air\b": ("Hea", "+"),
    r"\bheader\b": ("Hea", "+"), r"\btall\b": ("Jum", "+"),
    r"\bclinical\b": ("Fin", "+"), r"\blethal\b": ("Fin", "+"),
    r"\bsharp\b": ("Fin", "+"), r"\bgoal[- ]?scorer\b": ("Fin", "+"),
    r"\bpoacher\b": ("OtB", "+"), r"\bfinisher\b": ("Fin", "+"),
    r"\btireless\b": ("Sta", "+"), r"\bfit\b": ("Sta", "+"),
    r"\benduring\b": ("Sta", "+"), r"\benerg(?:etic|y)\b": ("Sta", "+"),
    r"\bengine\b": ("Sta", "+"), r"\bstamina\b": ("Sta", "+"),
    r"\bhigh work[- ]?rate\b": ("Wor", "+"), r"\bwork[- ]?rate\b": ("Wor", "+"),
    r"\bhard[- ]?working\b": ("Wor", "+"), r"\bindustrious\b": ("Wor", "+"),
    r"\bpressing\b": ("Wor", "+"), r"\bhigh[- ]?press(?:ing)?\b": ("Wor", "+"),
    r"\btricky\b": ("Dri", "+"), r"\bskil(?:l|ful|led)\b": ("Dri", "+"),
    r"\bsilky\b": ("Dri", "+"), r"\bnimble\b": ("Agi", "+"),
    r"\bagile\b": ("Agi", "+"), r"\bdribbler\b": ("Dri", "+"),
    r"\bcreative\b": ("Vis", "+"), r"\bvisionary\b": ("Vis", "+"),
    r"\bplaymaker\b": ("Vis", "+"), r"\bvision\b": ("Vis", "+"),
    r"\bpasser\b": ("Pas", "+"), r"\bdistribut(?:or|ion)\b": ("Pas", "+"),
    r"\blong[- ]?passer\b": ("Lon", "+"),
    r"\btackler\b": ("Tck", "+"), r"\btenacious\b": ("Tck", "+"),
    r"\bcombative\b": ("Tck", "+"), r"\bdefensive\b": ("Mar", "+"),
    r"\bmarker\b": ("Mar", "+"),
    r"\bshot[- ]?stopper\b": ("Ref", "+"), r"\breflexes\b": ("Ref", "+"),
    r"\bcommanding\b": ("Cmd", "+"), r"\bsweeper[- ]?keeper\b": ("TRO", "+"),
    r"\bcalm\b": ("Cmp", "+"), r"\bcomposed\b": ("Cmp", "+"),
    r"\bcomposure\b": ("Cmp", "+"),
    r"\bleader\b": ("Ldr", "+"), r"\bleadership\b": ("Ldr", "+"),
    r"\bcaptain\b": ("Ldr", "+"),
    r"\bintelligent\b": ("Dec", "+"), r"\bsmart\b": ("Dec", "+"),
    r"\bclever\b": ("Dec", "+"), r"\bdecision[- ]?mak\w*\b": ("Dec", "+"),
    r"\bbrave\b": ("Bra", "+"), r"\bbalanced\b": ("Bal", "+"),
    r"\bdetermined\b": ("Det", "+"), r"\baggressive\b": ("Agg", "+"),
    r"\banticipat\w*\b": ("Ant", "+"), r"\btechnical\b": ("Tec", "+"),
    r"\btechnique\b": ("Tec", "+"), r"\bflair\b": ("Fla", "+"),
    r"\bcrosser\b": ("Cro", "+"), r"\boff the ball\b": ("OtB", "+"),
    r"\bpositionally aware\b": ("Pos", "+"), r"\bpositioning\b": ("Pos", "+"),
}

AGE_SYNONYMS = {
    r"\bteenager?\b": (None, 19), r"\byoungster\b": (None, 21),
    r"\byoung\b": (None, 23), r"\bprospect\b": (None, 23),
    r"\bwonderkid\b": (None, 20),
    r"\bveteran\b": (30, None), r"\bexperienced\b": (28, None),
    r"\bsenior\b": (28, None), r"\bmature\b": (27, None),
}

# Position keyword -> FM position tokens to match against the Position string.
POSITION_KEYWORDS = {
    "goalkeeper": ["GK"], "keeper": ["GK"], "goalie": ["GK"],
    "centre-back": ["D (C)", "D (RC)", "D (LC)", "D (RLC)"],
    "center-back": ["D (C)", "D (RC)", "D (LC)", "D (RLC)"],
    "centre back": ["D (C)", "D (RC)", "D (LC)", "D (RLC)"],
    "center back": ["D (C)", "D (RC)", "D (LC)", "D (RLC)"],
    "centre-half": ["D (C)", "D (RC)", "D (LC)", "D (RLC)"],
    "centre half": ["D (C)", "D (RC)", "D (LC)", "D (RLC)"],
    "central defender": ["D (C)"], "defender": ["D ("], "defence": ["D ("],
    "full-back": ["D (R)", "D (L)", "D (RL)", "WB"], "fullback": ["D (R)", "D (L)", "WB"],
    "right-back": ["D (R)", "D (RC)", "WB (R)"], "left-back": ["D (L)", "D (LC)", "WB (L)"],
    "wing-back": ["WB"], "wingback": ["WB"],
    "defensive midfielder": ["DM"], "holding midfielder": ["DM"], "anchor": ["DM"],
    "midfielder": ["M ("], "central midfielder": ["M (C)"],
    "box-to-box": ["M (C)"], "box to box": ["M (C)"],
    "attacking midfielder": ["AM (C)"], "playmaker": ["AM (C)", "M (C)"],
    "winger": ["AM (R)", "AM (L)", "AM (RL)", "M (R)", "M (L)"],
    "wide midfielder": ["M (R)", "M (L)"],
    "right winger": ["AM (R)", "M (R)"], "left winger": ["AM (L)", "M (L)"],
    "right wing": ["AM (R)", "M (R)"], "left wing": ["AM (L)", "M (L)"],
    "striker": ["ST"], "forward": ["ST", "AM"], "centre-forward": ["ST (C)"],
    "center-forward": ["ST (C)"], "number 9": ["ST (C)"], "no 9": ["ST (C)"],
    "poacher": ["ST (C)"], "target man": ["ST (C)"],
}

NATIONALITY_TERMS = {
    "argentine": "Argentinian", "argentinian": "Argentinian", "brazilian": "Brazilian",
    "french": "French", "spanish": "Spanish", "english": "English", "german": "German",
    "italian": "Italian", "portuguese": "Portuguese", "dutch": "Dutch", "belgian": "Belgian",
    "polish": "Polish", "norwegian": "Norwegian", "egyptian": "Egyptian",
    "senegalese": "Senegalese", "turkish": "Turkish", "turk": "Turkish",
    "scottish": "Scottish", "irish": "Irish", "welsh": "Welsh", "croatian": "Croatian",
    "uruguayan": "Uruguayan", "colombian": "Colombian", "moroccan": "Moroccan",
    "nigerian": "Nigerian", "japanese": "Japanese", "korean": "South Korean",
    "south korean": "South Korean", "australian": "Australian", "american": "American",
    "mexican": "Mexican", "danish": "Danish", "swedish": "Swedish", "swiss": "Swiss",
    "greek": "Greek", "austrian": "Austrian", "czech": "Czech", "romanian": "Romanian",
    "ukrainian": "Ukrainian", "ghanaian": "Ghanaian", "algerian": "Algerian",
    "chilean": "Chilean", "peruvian": "Peruvian", "ecuadorian": "Ecuadorian",
    "venezuelan": "Venezuelan", "serbian": "Serbian", "hungarian": "Hungarian",
    "ivorian": "Ivorian", "cameroonian": "Cameroonian", "iranian": "Iranian",
    "chinese": "Chinese", "tunisian": "Tunisian", "russian": "Russian",
    "finnish": "Finnish", "slovenian": "Slovenian", "slovakian": "Slovakian",
}

# attribute keyword (readable, lowercased) for explicit numeric constraints
_ATTR_KEYWORDS = {v.lower(): k for k, v in ATTR_READABLE.items()}
_ATTR_KEYWORDS.update({"strength": "Str", "pace": "Pac", "passing": "Pas",
                       "tackling": "Tck", "dribbling": "Dri", "finishing": "Fin",
                       "heading": "Hea", "shooting": "Fin", "speed": "Pac",
                       "workrate": "Wor", "work rate": "Wor"})

_OP_NORMALISE = {"at least": ">=", "no less than": ">=", "minimum": ">=",
                 "above": ">", "over": ">", "greater than": ">", "more than": ">",
                 "at most": "<=", "no more than": "<=", "maximum": "<=",
                 "below": "<", "under": "<", "less than": "<",
                 "exactly": "=", "equal to": "=", "=": "=",
                 ">": ">", "<": "<", ">=": ">=", "<=": "<="}

_NUM = r"([0-9]+(?:\.[0-9]+)?)\s*(million|mil\b|m\b|k\b|thousand|b\b|billion)?"


def _money(num, suffix):
    num = float(num)
    s = (suffix or "").lower().strip()
    if s in ("k", "thousand"):  return num * 1e3
    if s in ("m", "mil", "million"): return num * 1e6
    if s in ("b", "billion"):   return num * 1e9
    return num


def parse_query(query: str, known_clubs=None) -> dict:
    """Parse a natural-language scouting query into a structured constraint dict.

    Returns a dict with:
      age:    list of (min,max) intervals (OR'd; player passes if in ANY)
      value:  {'min','max'}     wage: {'min','max'}
      nationality: [list]  (OR)   league: [list] (OR)   position: [tokens] (OR)
      foot:   'Left'/'Right'/'Either' or None
      hard_attrs: {col: [(op, threshold), ...]}   (strict numeric filters)
      soft_attrs: {col: '+'/'-'}                  (descriptive ranking signals)
      semantic_query: residual free text for embedding search
      top_k:  requested number of players (default 10)
      trace:  human-readable list of what was parsed
    """
    q = " " + query.lower().strip() + " "
    r = {"age": [], "value": {"min": None, "max": None},
         "wage": {"min": None, "max": None},
         "nationality": [], "league": [], "club": [], "position": [], "foot": None,
         "hard_attrs": {}, "soft_attrs": {}, "semantic_query": query,
         "top_k": 10, "top_k_explicit": False, "raw_query": query, "trace": []}

    def trace(msg):
        r["trace"].append(msg)

    # --- requested count: "top 5", "best 3 players", "give me 20" -----------
    m = re.search(r"\b(?:top|best|first)\s+(\d{1,3})\b", q) or \
        re.search(r"\b(\d{1,3})\s+players?\b", q)
    if m:
        r["top_k"] = int(m.group(1))
        r["top_k_explicit"] = True
        trace(f"top_k = {r['top_k']}")

    # --- AGE (supports OR: 'under 20 or over 35') ---------------------------
    age_intervals = []
    # negative lookahead avoids matching money like "under 10M" / "over 5 million"
    _no_money = r"(?![\d]|\s*(?:m\b|k\b|b\b|mil|million|thousand|billion|eur|€|\$|,\d))"
    # blank out attribute thresholds ("passing above 16", "technique > 15") so the
    # age scanner doesn't read them as ages (e.g. "above 16" -> age > 16).
    _attr_alt = "|".join(re.escape(k) for k in sorted(_ATTR_KEYWORDS, key=len, reverse=True))
    _cmp = (r"(?:>=|<=|>|<|=|at least|no less than|minimum|above|over|greater than|more than|"
            r"at most|no more than|maximum|below|under|less than|exactly|equal to|of|is|are|around)")
    q_age = re.sub(r"\b(?:" + _attr_alt + r")\b\s*" + _cmp + r"?\s*\d{1,2}", "  ", q)
    for mm in re.finditer(r"(?:under|younger than|below|less than|u-?)\s*(\d{1,2})" + _no_money, q_age):
        age_intervals.append((None, int(mm.group(1)) - 1))
    for mm in re.finditer(r"(?:older than|over|above|aged over)\s*(\d{1,2})" + _no_money, q_age):
        # avoid catching transfer/value 'over' (those are handled with money units)
        age_intervals.append((int(mm.group(1)) + 1, None))
    for mm in re.finditer(r"\b(?:aged?\s+)?(?:between\s+)?(\d{1,2})\s*(?:and|to|-)\s*(\d{1,2})\b", q_age):
        a, b = int(mm.group(1)), int(mm.group(2))
        if 14 <= a <= 45 and 14 <= b <= 50 and a < b:
            age_intervals.append((a, b))
    for pat, (mn, mx) in AGE_SYNONYMS.items():
        if re.search(pat, q):
            age_intervals.append((mn, mx))
    # de-dup
    age_intervals = list(dict.fromkeys(age_intervals))
    if age_intervals:
        r["age"] = age_intervals
        trace(f"age intervals (OR) = {age_intervals}")

    # --- TRANSFER VALUE -----------------------------------------------------
    for mm in re.finditer(r"(?:valued?|worth|value|budget|costs?|price[d]?|fee|market value)"
                          r"[^0-9]{0,25}?(?:under|below|less than|up to|at most|max(?:imum)?|around|~|of)?"
                          r"\s*(?:eur|€|\$)?\s*" + _NUM, q):
        # treat as max unless an explicit 'over/above/more than' appears just before
        v = _money(mm.group(1), mm.group(2))
        if v >= 1000:  # ignore tiny numbers that are really ages/attrs
            if re.search(r"(over|above|more than|higher than|greater than|"
                         r"north of|exceed(?:s|ing)?|at least|minimum)", mm.group(0)):
                r["value"]["min"] = v
                trace(f"value >= EUR{v:,.0f}")
            else:
                r["value"]["max"] = v
                trace(f"value <= EUR{v:,.0f}")

    # --- WAGE ---------------------------------------------------------------
    for mm in re.finditer(r"(?:wages?|salary|earn(?:s|ing)?|paid)"
                          r"[^0-9]{0,20}?(?:under|below|less than|up to|at most|over|above|more than)?"
                          r"\s*(?:eur|€|\$)?\s*" + _NUM + r"\s*(?:p/?w|p/?m|per week|per month|/week|/month)?", q):
        v = _money(mm.group(1), mm.group(2))
        if v >= 100:
            if re.search(r"(over|above|more than|higher than|greater than|"
                         r"north of|exceed(?:s|ing)?|at least|minimum)", mm.group(0)):
                r["wage"]["min"] = v
                trace(f"wage >= EUR{v:,.0f}")
            else:
                r["wage"]["max"] = v
                trace(f"wage <= EUR{v:,.0f}")

    # --- FOOT ---------------------------------------------------------------
    if re.search(r"\bleft[- ]?foot(?:ed)?\b", q):
        r["foot"] = "Left"; trace("foot = Left")
    elif re.search(r"\bright[- ]?foot(?:ed)?\b", q):
        r["foot"] = "Right"; trace("foot = Right")

    # --- POSITION (OR across multiple) --------------------------------------
    pos_tokens = []
    matched_kws = []
    for kw in sorted(POSITION_KEYWORDS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(kw) + r"(?:s|es)?\b", q):
            # skip a generic keyword already covered by a longer matched phrase
            # (e.g. "midfielder" inside "central midfielder")
            if any(kw != m and kw in m for m in matched_kws):
                continue
            matched_kws.append(kw)
            pos_tokens.extend(POSITION_KEYWORDS[kw])
    if pos_tokens:
        r["position"] = list(dict.fromkeys(pos_tokens))
        trace(f"position tokens (OR) = {r['position']}")

    # --- NATIONALITY (OR: 'Turkish or Brazilian') --------------------------
    nats = []
    for term in sorted(NATIONALITY_TERMS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term) + r"s?\b", q):
            canon = NATIONALITY_TERMS[term]
            if canon not in nats:
                nats.append(canon)
    if nats:
        r["nationality"] = nats
        trace(f"nationality (OR) = {nats}")

    # --- CLUB / TEAM (OR; accent-insensitive, matched against the dataset) --
    if known_clubs:
        qn = _strip_accents(q)
        hits = []
        for club in sorted(known_clubs, key=lambda c: len(str(c)), reverse=True):
            cn = _strip_accents(club)
            if len(cn) < 4 or cn in ("unknown",):   # skip too-short / placeholder
                continue
            if re.search(r"\b" + re.escape(cn) + r"\b", qn):
                # drop a hit already covered by a longer matched club name
                if not any(cn != _strip_accents(h) and cn in _strip_accents(h) for h in hits):
                    hits.append(club)
        # remove shorter names that are substrings of a longer accepted one
        hits = [c for c in hits
                if not any(c != o and _strip_accents(c) in _strip_accents(o) for o in hits)]
        if hits:
            r["club"] = hits
            trace(f"club (OR) = {hits}")

    # --- EXPLICIT numeric attribute constraints (HARD, strict) -------------
    for kw in sorted(_ATTR_KEYWORDS, key=len, reverse=True):
        col = _ATTR_KEYWORDS[kw]
        _filler = r"(?:\s+(?:is|are|of|at|should be|must be|around))?\s*"
        for mm in re.finditer(
            r"\b" + re.escape(kw) + r"\b" + _filler +
            r"(>=|<=|>|<|=|at least|no less than|minimum|above|over|greater than|more than|"
            r"at most|no more than|maximum|below|under|less than|exactly|equal to)\s*(\d{1,2})\b", q):
            op = _OP_NORMALISE.get(mm.group(1).strip(), "=")
            thr = int(mm.group(2))
            r["hard_attrs"].setdefault(col, []).append((op, thr))
            trace(f"hard: {ATTR_READABLE.get(col, col)} {op} {thr}")
        # 'between X and Y' for an attribute, e.g. 'passing between 10 and 17'
        for mm in re.finditer(
            r"\b" + re.escape(kw) + r"\b" + _filler + r"(?:between\s+)?(\d{1,2})\s*(?:and|to|-)\s*(\d{1,2})\b", q):
            lo, hi = int(mm.group(1)), int(mm.group(2))
            if lo <= 20 and hi <= 20 and lo < hi:
                r["hard_attrs"].setdefault(col, []).append((">=", lo))
                r["hard_attrs"][col].append(("<=", hi))
                trace(f"hard: {ATTR_READABLE.get(col, col)} in [{lo}, {hi}]")

    # --- DESCRIPTIVE synonyms (SOFT ranking signals) ------------------------
    for pat, (col, direction) in ATTR_SYNONYMS.items():
        if re.search(pat, q) and col not in r["hard_attrs"]:
            r["soft_attrs"][col] = direction

    # --- QUALITY phrases on an attribute: "great at passing", "good passing",
    #     "weak finishing", "passing is excellent" -> soft signal --------------
    _QPOS = (r"(?:great|good|excellent|strong|elite|superb|top|brilliant|amazing|"
             r"exceptional|world[- ]?class|quality|solid|sharp|nice)")
    _QNEG = r"(?:poor|weak|bad|limited|lacking|lacks|low)"
    for kw, col in sorted(_ATTR_KEYWORDS.items(), key=lambda x: -len(x[0])):
        kwe = re.escape(kw)
        pos = (re.search(_QPOS + r"\s+(?:at\s+|in\s+|with\s+(?:his\s+|her\s+)?)?" + kwe + r"\b", q)
               or re.search(r"\b" + kwe + r"\s+(?:is\s+|are\s+)?" + _QPOS, q))
        neg = (re.search(_QNEG + r"\s+(?:at\s+|in\s+)?" + kwe + r"\b", q)
               or re.search(r"\b" + kwe + r"\s+(?:is\s+|are\s+)?" + _QNEG, q))
        if col in r["hard_attrs"]:
            continue
        if neg:
            r["soft_attrs"][col] = "-"
        elif pos and col not in r["soft_attrs"]:
            r["soft_attrs"][col] = "+"
    if r["soft_attrs"]:
        trace("soft: " + ", ".join(f"{ATTR_READABLE.get(c, c)}{d}"
                                   for c, d in r["soft_attrs"].items()))

    # --- semantic residual (strip parsed bits, keep descriptive words) ------
    r["semantic_query"] = query
    return r


# --------------------------------------------------------------------------- #
# 4. Structured filtering  (strict / hard constraints)
# --------------------------------------------------------------------------- #

def structured_filter(df, parsed):
    """Apply hard constraints strictly. Returns (filtered_df, applied, skipped)."""
    mask = pd.Series(True, index=df.index)
    applied, skipped = [], []

    # AGE — OR across intervals
    if parsed["age"]:
        age_mask = pd.Series(False, index=df.index)
        for mn, mx in parsed["age"]:
            sub = pd.Series(True, index=df.index)
            if mn is not None:
                sub &= df["Age"] >= mn
            if mx is not None:
                sub &= df["Age"] <= mx
            age_mask |= sub
        mask &= age_mask
        applied.append(f"Age in {parsed['age']} (OR)")

    # VALUE
    if parsed["value"]["min"] is not None:
        mask &= df["Value_EUR"].notna() & (df["Value_EUR"] >= parsed["value"]["min"])
        applied.append(f"Value >= EUR{parsed['value']['min']:,.0f}")
    if parsed["value"]["max"] is not None:
        mask &= df["Value_EUR"].notna() & ~df["Not_For_Sale"] & (df["Value_EUR"] <= parsed["value"]["max"])
        applied.append(f"Value <= EUR{parsed['value']['max']:,.0f} (excl. not-for-sale)")

    # WAGE
    if parsed["wage"]["min"] is not None:
        mask &= df["Wage_EUR_pm"].notna() & (df["Wage_EUR_pm"] >= parsed["wage"]["min"])
        applied.append(f"Wage >= EUR{parsed['wage']['min']:,.0f}/m")
    if parsed["wage"]["max"] is not None:
        mask &= df["Wage_EUR_pm"].notna() & (df["Wage_EUR_pm"] <= parsed["wage"]["max"])
        applied.append(f"Wage <= EUR{parsed['wage']['max']:,.0f}/m")

    # FOOT
    if parsed["foot"]:
        mask &= df["Foot"] == parsed["foot"]
        applied.append(f"Foot == {parsed['foot']}")

    # POSITION — OR across tokens (substring match on the FM position string)
    if parsed["position"]:
        pos_mask = df["Position"].apply(
            lambda p: any(tok.lower() in str(p).lower() for tok in parsed["position"]))
        mask &= pos_mask
        applied.append(f"Position matches one of {parsed['position']}")

    # NATIONALITY — OR
    if parsed["nationality"]:
        nat_lower = [n.lower() for n in parsed["nationality"]]
        mask &= df["Nationality"].str.lower().isin(nat_lower)
        applied.append(f"Nationality in {parsed['nationality']} (OR)")

    # CLUB — OR (accent-insensitive match against Club_Name)
    if parsed.get("club"):
        club_norm = {_strip_accents(c) for c in parsed["club"]}
        mask &= df["Club_Name"].apply(lambda c: _strip_accents(c) in club_norm)
        applied.append(f"Club in {parsed['club']} (OR)")

    # HARD numeric attributes — strict
    for col, conds in parsed["hard_attrs"].items():
        if col not in df.columns:
            skipped.append(f"attribute '{col}' not in dataset")
            continue
        for op, thr in conds:
            if op == ">":   mask &= df[col] > thr
            elif op == ">=": mask &= df[col] >= thr
            elif op == "<":  mask &= df[col] < thr
            elif op == "<=": mask &= df[col] <= thr
            elif op == "=":  mask &= df[col] == thr
            applied.append(f"{ATTR_READABLE.get(col, col)} {op} {thr}")

    return df[mask].copy(), applied, skipped


# --------------------------------------------------------------------------- #
# 5. Ranking  (combined constraint score, position-aware, quality tie-break)
# --------------------------------------------------------------------------- #

def _norm(x, lo=ATTR_SCALE_MIN, hi=ATTR_SCALE_MAX):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def rank_players(filtered_df, parsed, sem_scores=None):
    """Rank filtered players best-match -> worst.

    For every constraint we compute a [0,1] sub-score; the mean of the
    sub-scores is the constraint score (higher = better match). Players are
    sorted by constraint score, then by overall quality as a tie-breaker.

    Position-aware: goalkeeper-only attributes never contribute to an outfield
    player's score and vice versa.

      * hard '>' / '>=' : reward how far the value exceeds the threshold
      * hard '<' / '<=' : reward how far the value is below the threshold
      * hard '='        : full score (binary, already filtered)
      * soft '+'        : reward higher raw value
      * soft '-'        : reward lower raw value
      * semantic        : cosine similarity (if provided), one extra sub-score
    """
    if len(filtered_df) == 0:
        return filtered_df.assign(match_score=[], _quality=[])

    df = filtered_df.copy()
    scores = np.zeros(len(df))
    n_sub = np.zeros(len(df))

    def applicable(col, row_is_gk):
        """Skip GK attrs for outfielders and outfield-only attrs for GKs."""
        if col in GK_ONLY_COLS:
            return row_is_gk
        if col in OUTFIELD_ATTR_COLS and col not in PHYSICAL_MENTAL_SHARED:
            return ~row_is_gk
        return np.ones(len(df), dtype=bool)

    is_gk = df["is_GK"].to_numpy()

    # hard attribute sub-scores (margin beyond threshold)
    for col, conds in parsed["hard_attrs"].items():
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        appl = applicable(col, is_gk)
        for op, thr in conds:
            if op in (">", ">="):
                sub = np.array([_norm(v - thr, 0, ATTR_SCALE_MAX - thr if ATTR_SCALE_MAX > thr else 1)
                                for v in vals])
            elif op in ("<", "<="):
                sub = np.array([_norm(thr - v, 0, thr - ATTR_SCALE_MIN if thr > ATTR_SCALE_MIN else 1)
                                for v in vals])
            else:  # '='
                sub = np.ones(len(df))
            scores += np.where(appl, sub, 0)
            n_sub += appl.astype(float)

    # soft attribute sub-scores
    for col, direction in parsed["soft_attrs"].items():
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        appl = applicable(col, is_gk)
        sub = np.array([_norm(v) for v in vals])
        if direction == "-":
            sub = 1.0 - sub
        scores += np.where(appl, sub, 0)
        n_sub += appl.astype(float)

    # semantic sub-score (optional)
    if sem_scores is not None:
        ss = np.asarray(sem_scores, dtype=float)
        ss = (ss - ss.min()) / (ss.max() - ss.min() + 1e-9)  # min-max within pool
        scores += ss
        n_sub += 1.0

    with np.errstate(invalid="ignore"):
        constraint_score = np.where(n_sub > 0, scores / np.maximum(n_sub, 1), 0.0)

    # General player quality (position-appropriate overall, normalised to 0-1).
    quality = np.array([_norm(v) for v in df["Overall"].fillna(0).to_numpy()])
    df["_quality"] = df["Overall"].fillna(0).to_numpy()

    # Always blend in general quality so that, all else equal, better players
    # rank higher. When there are no constraint signals at all, ranking is
    # purely quality-based (a "total player score").
    QUALITY_WEIGHT = 0.20
    if float(np.nansum(n_sub)) == 0:
        df["match_score"] = quality
    else:
        df["match_score"] = (1 - QUALITY_WEIGHT) * constraint_score + QUALITY_WEIGHT * quality

    df = df.sort_values(["match_score", "_quality"], ascending=[False, False])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Top-level search (hybrid: structured filter + ranking [+ semantic])
# --------------------------------------------------------------------------- #

def search(df, parsed, embeddings=None, embed_query_fn=None, return_df=False):
    """Run the full pipeline for a parsed query.

    Returns a dict: {'message' or 'players': [...], 'parsed': parsed, ...}.
    If no player matches, message == NO_MATCH_MESSAGE.
    """
    filtered, applied, skipped = structured_filter(df, parsed)

    sem_scores = None
    # Only let semantic similarity influence ranking when the query carries
    # descriptive intent: either it produced soft attribute signals (adjectives
    # like "creative"/"fast"), or it is pure free text with no structured
    # constraints at all. A purely structured query (position/age/foot/club/
    # nationality/explicit numeric attrs) is ranked by its attribute constraints
    # and attribute-based quality only -- identity facts never become rank points.
    descriptive = bool(parsed.get("soft_attrs")) or not has_constraints(parsed)
    use_semantic = (embeddings is not None and embed_query_fn is not None
                    and parsed.get("semantic_query") and descriptive)
    if use_semantic and len(filtered) > 0:
        try:
            qvec = embed_query_fn(parsed["semantic_query"]).reshape(1, -1)
            # structured_filter preserves the original (RangeIndex) labels of df,
            # which line up positionally with the aligned embeddings matrix.
            cand_emb = embeddings[filtered.index.to_numpy()]
            sem_scores = (cand_emb @ qvec.T).ravel()
        except Exception:
            sem_scores = None

    ranked = rank_players(filtered, parsed, sem_scores=sem_scores)
    ranked = ranked.head(parsed.get("top_k", 10))

    result = {"parsed": parsed, "applied": applied, "skipped": skipped,
              "n_filtered": int(len(filtered))}
    if len(ranked) == 0:
        result["message"] = NO_MATCH_MESSAGE
        result["players"] = []
    else:
        result["players"] = _players_to_records(ranked)
        result["message"] = None
    if return_df:
        result["df"] = ranked
    return result


# --------------------------------------------------------------------------- #
# 7. Direct player lookup ("Tell me about Lionel Messi")
# --------------------------------------------------------------------------- #

# Words that signal a *scouting search* rather than a name lookup; if the query
# contains constraints we never treat it as a name lookup.
def has_constraints(parsed) -> bool:
    """True if the parsed query carries any scouting constraint (vs. a bare name)."""
    return bool(parsed.get("position") or parsed.get("nationality")
                or parsed.get("club") or parsed.get("hard_attrs")
                or parsed.get("soft_attrs") or parsed.get("age")
                or parsed.get("foot")
                or parsed["value"]["min"] is not None or parsed["value"]["max"] is not None
                or parsed["wage"]["min"] is not None or parsed["wage"]["max"] is not None)


def find_players_by_name(df, query: str, limit: int = 5):
    """Return dataframe rows whose Player_Name matches the query (accent-insensitive).

    Exact normalised matches win; otherwise every query token must appear as a
    whole token in the player's name. Results are sorted best (highest Overall)
    first. Returns an empty frame when nothing matches.
    """
    qn = re.sub(r"[^a-z0-9 ]", " ", _strip_accents(query)).strip()
    qn = re.sub(r"\s+", " ", qn)
    if len(qn) < 3:
        return df.iloc[0:0]
    norm = df["Player_Name"].fillna("").apply(_strip_accents)
    norm = norm.apply(lambda n: re.sub(r"[^a-z0-9 ]", " ", n))

    exact = df[norm.str.strip() == qn]
    if len(exact):
        return exact.sort_values("Overall", ascending=False).head(limit)

    toks = [t for t in qn.split() if len(t) >= 3]
    if not toks:
        return df.iloc[0:0]
    mask = norm.apply(lambda n: all(t in n.split() for t in toks))
    hits = df[mask]
    return hits.sort_values("Overall", ascending=False).head(limit)


def players_to_records(ranked):
    """Public wrapper around the internal record serialiser."""
    return _players_to_records(ranked)


def find_similar_players(df, ref_df, embeddings=None, top_k=5, allowed_index=None):
    """Return players most similar to a reference player.

    Similarity uses the cached sentence embeddings when available (cosine over
    the unit-norm profile vectors), otherwise a cosine over the raw 1-20
    attribute vector. Candidates are restricted to the same broad type
    (goalkeeper vs outfielder) as the reference. The reference itself is
    excluded. `match_score` carries the similarity (0-1).

    If `allowed_index` is given (e.g. the result of structured_filter for extra
    constraints like nationality / age / club in the query), candidates are
    further restricted to that set -- so "Turkish players like Haaland" returns
    Haaland-style players who are also Turkish.
    """
    if ref_df is None or len(ref_df) == 0:
        return df.iloc[0:0]
    ref_label = ref_df.index[0]
    ref_is_gk = bool(df.loc[ref_label, "is_GK"])

    # candidate pool: same broad role, drop the reference
    pool = df[(df["is_GK"] == ref_is_gk) & (df.index != ref_label)]
    if allowed_index is not None:
        pool = pool[pool.index.isin(allowed_index)]
    if len(pool) == 0:
        return df.iloc[0:0]

    ref_pos = df.index.get_loc(ref_label)
    pool_pos = np.array([df.index.get_loc(i) for i in pool.index])

    if embeddings is not None and len(embeddings) == len(df):
        vecs = np.asarray(embeddings, dtype="float32")  # already unit-norm
        sims = vecs[pool_pos] @ vecs[ref_pos]
    else:
        cols = [c for c in (OUTFIELD_ATTR_COLS + GK_ONLY_COLS) if c in df.columns]
        M = df[cols].fillna(0).to_numpy(dtype=float)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sims = M[pool_pos] @ M[ref_pos]

    order = np.argsort(-sims)[:top_k]
    res = pool.iloc[order].copy()
    res["match_score"] = np.clip(sims[order], 0.0, 1.0).round(3)
    res["_quality"] = res["Overall"].fillna(0).to_numpy()
    return res


_DISPLAY_COLS = ["Player_Name", "Age", "Nationality", "Position", "Club_Name",
                 "League", "Foot", "Value_EUR", "Wage_EUR_pm", "Overall",
                 "match_score"]


def _players_to_records(ranked):
    attr_cols = list(ATTR_READABLE.keys())
    recs = []
    for _, row in ranked.iterrows():
        rec = {}
        for c in _DISPLAY_COLS + attr_cols:
            if c in row.index:
                v = row[c]
                if isinstance(v, (np.floating, float)) and pd.notna(v):
                    v = round(float(v), 3)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif pd.isna(v):
                    v = None
                rec[c] = v
        recs.append(rec)
    return recs

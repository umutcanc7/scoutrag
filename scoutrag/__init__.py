"""ScoutRAG: constrained hybrid retrieval over structured and semantic player attributes.

Data source note: all player attributes come from a Football Manager 2024 export.
These are designer-set game attributes, NOT real-world scouting data. The project
uses them as a synthetic / prototype tabular dataset for validating the retrieval
and grounding pipeline.
"""

from .core import (
    ATTR_READABLE,
    OUTFIELD_ATTR_COLS,
    GK_ONLY_COLS,
    load_and_clean,
    align_embeddings,
    build_profiles,
    parse_query,
    structured_filter,
    rank_players,
    search,
    find_players_by_name,
    find_similar_players,
    players_to_records,
    has_constraints,
    NO_MATCH_MESSAGE,
)

__all__ = [
    "ATTR_READABLE",
    "OUTFIELD_ATTR_COLS",
    "GK_ONLY_COLS",
    "load_and_clean",
    "align_embeddings",
    "build_profiles",
    "parse_query",
    "structured_filter",
    "rank_players",
    "search",
    "find_players_by_name",
    "find_similar_players",
    "players_to_records",
    "has_constraints",
    "NO_MATCH_MESSAGE",
]

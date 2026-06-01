"""Rebuild player_embeddings.pkl for ScoutRAG.

Encodes each player's *play-style* text (scoutrag.build_profiles -> 'embed_text':
position + foot + style + 1-20 attributes, with no name / nationality / age) using
the same MiniLM model the backend and notebook use, and writes the cache in the
format scoutrag.align_embeddings expects: a single array indexed by the ORIGINAL
CSV row (raw_idx), so cleaned rows line up after filtering.

Run once after deleting the old cache (or whenever embed_text changes):
    cd "CS 455 Project"
    pip install -r requirements.txt
    python build_embeddings.py

Then restart the backend; the header should show "sem on".
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from scoutrag import load_and_clean, build_profiles

CSV_PATH = "fmdata24llm.csv"
PKL_PATH = "player_embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    print("Loading and cleaning dataset ...")
    df = build_profiles(load_and_clean(CSV_PATH, verbose=True))
    print(f"  clean rows: {len(df):,} | raw_idx max: {int(df['raw_idx'].max()):,}")

    print(f"Loading embedding model '{MODEL_NAME}' (downloads once) ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding play-style profiles ...")
    vecs = model.encode(
        df["embed_text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    # Place each clean row's vector at its ORIGINAL CSV position so the cache can
    # be indexed by raw_idx. Removed/invalid rows keep a zero vector (never read,
    # because they are dropped before embeddings are ever consulted).
    dim = vecs.shape[1]
    n_raw = int(df["raw_idx"].max()) + 1
    full = np.zeros((n_raw, dim), dtype="float32")
    full[df["raw_idx"].to_numpy()] = vecs

    with open(PKL_PATH, "wb") as f:
        pickle.dump({"embeddings": full, "model": MODEL_NAME, "dim": dim}, f)
    print(f"Saved {PKL_PATH}: {full.shape} (raw-indexed). Done.")


if __name__ == "__main__":
    main()

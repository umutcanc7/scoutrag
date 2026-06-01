# ScoutRAG — Constrained Hybrid Retrieval for Football Player Scouting

CS455 (Large Language Models) term project · standard track.
Çubukçu · Özer · Yılmaz

ScoutRAG is a retrieval-augmented scouting assistant. You write a natural-language query
("a clinical, left-footed striker under 21 valued below €10M") and it returns the players that
match, ranked best→worst. It combines **strict symbolic filtering** for precise constraints,
**descriptive-term → constraint** mapping via a local LLM, **semantic re-ranking** with sentence
embeddings, and a **post-generation grounding check** that verifies cited numbers against the
retrieved rows.

> **Dataset honesty note.** The data is a Football Manager 2024 export. Every attribute is a
> *designer-set game value on a 1–20 scale* — **not** real-world scouting data. We treat it as a
> synthetic / prototype tabular dataset used to validate the retrieval and grounding pipeline.

---

## What's in this repo

| Path | What it is |
|---|---|
| `ScoutRAG_CS455.ipynb` | Main notebook: cleaning, EDA, search/ranking, LLM, grounding, evaluation |
| `scoutrag/core.py` | Engine: cleaning, query parsing, strict filtering, position-aware ranking |
| `scoutrag/llm.py` | Local LLM (Ollama) query rewriting, grounded report, grounding verification |
| `backend/app.py` | FastAPI server wrapping the engine (`/search`, `/health`) |
| `docs/index.html` | Static green chat UI (GitHub-Pages ready) that calls the backend |
| `fmdata24llm.csv` | FM24 dataset |
| `player_embeddings.pkl` | Cached MiniLM embeddings (rebuilt automatically if missing) |
| `eda_outputs/` | Saved EDA figures for the report |
| `requirements.txt` | Python dependencies |
| `ScoutRAG_CS455_colab_backup.ipynb` | The original Colab notebook, kept for reference |

The notebook, the backend, and the UI all share **one engine** (`scoutrag/`), so behaviour is
identical everywhere.

---

## Hardware & software requirements

Built and tested for **Apple Silicon (M-series), 16 GB RAM** — comfortable.

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.10+ | |
| RAM | 8 GB min, 16 GB recommended | Dataset + embeddings fit easily |
| GPU | none required | Embeddings run on CPU/MPS; ~40k rows encode in well under a minute |
| Semantic search | `sentence-transformers` | Downloads `all-MiniLM-L6-v2` (~80 MB) once |
| Local LLM (optional) | [Ollama](https://ollama.com) + `llama3.1` (~4.9 GB) | Any local model works — set `DEFAULT_MODEL` in `scoutrag/llm.py` to match. Needs ~8 GB free RAM while running. **Everything works without it** — the rule-based parser handles many scouting terms and a deterministic report template is used. |
| `faiss-cpu` | optional | sklearn / NumPy cosine is the automatic fallback (used by default on Apple Silicon) |

Nothing here uses Google Drive, Colab, or any paid API.

---

## Setup (local, one time)

```bash
cd "CS 455 Project"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional — enable the local LLM:

```bash
brew install ollama          # or download from ollama.com
ollama serve                 # starts the local server on :11434
ollama pull llama3.1
```

---

## Run the notebook

```bash
jupyter notebook ScoutRAG_CS455.ipynb
```

Run top to bottom. It cleans the data, does EDA (figures saved to `eda_outputs/`), demonstrates
the search/ranking module, the grounding check, and a 40-query evaluation with baselines.

---

## Run the web app

**1. Start the backend** (serves the engine on `http://localhost:8000`):

```bash
uvicorn backend.app:app --reload --port 8000
```

First start loads the dataset and embeddings (a few seconds). Check it:
`http://localhost:8000/health`.

**2. Open the UI.** Open `docs/index.html` in a browser (double-click, or serve it):

```bash
python3 -m http.server 5500 --directory docs   # then visit http://localhost:5500
```

The UI's ⚙ settings let you set the backend URL, result count, LLM rewriting, and whether to
generate a grounded report.

### GitHub Pages deployment

Push the repo and enable **Settings → Pages → Deploy from branch → `/docs`**. The static UI will
be served at `https://<user>.github.io/<repo>/`. Because GitHub Pages is static-only, the page
still talks to a backend you run locally — open the UI's settings and point *Backend URL* at your
machine (`http://localhost:8000`). For a fully self-contained online demo you would host the
FastAPI backend on a small server and set that URL instead.

---

## How the engine works

```
query
  └─ (optional) Ollama rewrite:  "clinical visionary playmaker"
                                 → "playmaker, finishing >= 15, vision >= 15"
  └─ parse_query        → hard constraints (strict)  +  soft signals (ranking)
  └─ structured_filter  → keep only rows satisfying every hard constraint
  └─ rank_players       → combined, position-aware score; overall quality breaks ties
  └─ (optional) semantic re-rank within the filtered pool
  └─ top-k  (or  "There is no player like that in our database.")
```

- **Every attribute is usable as a constraint**, with `>`, `>=`, `<`, `<=`, `=`, and `between`
  ranges; plus age, nationality, league, position, foot, value, and wage.
- **OR conditions** are supported: `"Turkish or Brazilian"`, `"under 20 or over 35"`, multiple
  positions.
- **Precise numeric constraints are enforced strictly**; vague terms become ranking signals (and
  are upgraded to explicit constraints by the LLM when available).
- **Ranking is fair and position-aware**: for `passing > 15`, passing 17 outranks passing 15;
  goalkeeper attributes never rank outfielders and vice-versa.
- **Grounding verification** flags any player name or stat in a generated report that is not
  present in the retrieved rows.

---

## Limitations (honest)

- FM24 ratings are designer-set game values, not validated real-world ability.
- Descriptive thresholds (what counts as "strong") are heuristic design choices.
- The rule-based parser can miss rare phrasings; the LLM rewrite mitigates but is best-effort.
- "value" and "wage" are missing for some lower-league / not-for-sale players, so budget filters
  exclude rows with unknown values by design.

## Planned modules (not implemented yet)
1. **Player info** — `"Tell me about Lionel Messi"` → grounded single-player profile.
2. **Player comparison** — compare two players attribute-by-attribute.

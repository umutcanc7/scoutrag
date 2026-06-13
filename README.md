---
title: ScoutRAG
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
license: mit
---

# ScoutRAG — Constrained Hybrid Retrieval for Football Player Scouting

**CS455 (Large Language Models), Spring 2025/2026 — standard track term project.**
Mustafa Ege Özer · Umut Can Çubukçu · Yiğit Onur Yılmaz

ScoutRAG is a retrieval-augmented scouting assistant. You write a natural-language query —
*"a clinical, left-footed striker under 21 valued below €10M"* — and it returns the players that
match, ranked best → worst. It combines **strict symbolic filtering** for precise constraints, a
**descriptive-term → attribute** mapping via a local LLM, **semantic re-ranking** with sentence
embeddings, and a **post-generation grounding check** that verifies every cited number against
the retrieved rows.

> **Dataset honesty note (read this first).** The data is a Football Manager 2024 export. Every
> attribute is a *designer-set game value on a 1–20 scale* — **not** real-world scouting data. We
> treat it strictly as a **synthetic / game-derived prototype tabular dataset** used to validate the
> retrieval and grounding pipeline. We do **not** claim it as a real scouting dataset, and the
> system is **not** Text-to-SQL — there is no executable SQL generation. The honest framing is
> *constrained hybrid structured-symbolic + semantic retrieval over tabular player data*, and the
> most interesting contribution is the **post-generation grounding verification** step.

> **Data source & acquisition (CS455 §8 compliance).** Football Manager 2024 includes a
> built-in, officially supported scouting-view export command that writes selected columns to an
> RTF file on disk. We used this in-game feature — no web scraping, no third-party tools, no
> reverse engineering. The RTF was then converted to CSV via `convert_fm_table.py` (included in
> the repo). Because the export is an official, documented game feature (not a Terms-of-Service
> violation), no special permission was required beyond owning a licensed copy of the game. All
> player attributes in the resulting CSV are synthetic designer-set values, not personal data, so
> no anonymization or data-protection obligation applies. The conversion script and the resulting
> `fmdata24llm.csv` are both included in the submission for full reproducibility.

---

## 1. What's in this repo

| Path | What it is |
|---|---|
| `ScoutRAG_CS455.ipynb` | **Main notebook**: cleaning, EDA, search/ranking, LLM, grounding, evaluation |
| `ScoutRAG_Tests_and_Ablations.ipynb` | **Tests & ablations**: test-query gallery, baseline metrics, ablation studies, parser regression checks |
| `scoutrag/core.py` | Engine: cleaning, query parsing, strict filtering, position-aware ranking, semantic re-rank |
| `scoutrag/llm.py` | Local LLM (Ollama): intent routing, query rewriting, grounded report, grounding verification |
| `backend/app.py` | FastAPI server wrapping the engine (`/search`, `/health`) |
| `docs/index.html` | Static green chat UI (GitHub-Pages ready) that calls the backend |
| `build_embeddings.py` | Standalone script to (re)build the embedding cache |
| `fmdata24llm.csv` | FM24 dataset (~42.5k raw rows) |
| `player_embeddings.pkl` | Cached MiniLM embeddings — **git-ignored**, rebuilt automatically on first backend start |
| `eda_outputs/` | Saved EDA figures + summary stats for the report |
| `requirements.txt` | Python dependencies |
| `ScoutRAG_CS455_colab_backup.ipynb` | Colab-flavoured copy of the notebook, kept for reference |

The notebook, the backend, and the UI all share **one engine** (`scoutrag/`), so behaviour is
identical everywhere.

---

## 2. Getting the project (for teammates pulling the repo)

```bash
git clone https://github.com/umutcanc7/scoutrag.git
cd scoutrag
```

Already cloned? Pull the latest:

```bash
git pull origin main
```

> `player_embeddings.pkl` is **not** in the repo (it's large and git-ignored). You don't need to
> download it — the backend rebuilds it automatically the first time it starts (see §4), or you can
> build it manually (§6).

---

## 3. One-time setup

Requirements: **Python 3.10+**, ~8 GB RAM (16 GB comfortable), no GPU needed. Built and tested on
Apple Silicon; Linux/Colab work too. Nothing here uses Google Drive or a paid API.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Optional — enable the local LLM (Ollama).** Everything works without it (a rule-based parser and
a deterministic report template take over), but the LLM improves intent routing and descriptive-term
handling.

```bash
# macOS
brew install ollama
# or download the installer from https://ollama.com  (Windows / Linux)

ollama serve            # starts the local API on http://localhost:11434
ollama pull llama3.1    # ~4.9 GB, downloads once
```

Use a different model? Set `DEFAULT_MODEL` in `scoutrag/llm.py` to match.

---

## 4. Run the web app (backend + frontend)

**Step 1 — start the backend** (loads the dataset and serves the engine on `:8000`):

```bash
uvicorn backend.app:app --reload --port 8000
```

The **first** start downloads MiniLM (~80 MB) and builds `player_embeddings.pkl` once
(a minute or two on CPU). Later starts just load the cache. Confirm it's healthy at
<http://localhost:8000/health> — you want `"semantic": true` and, if Ollama is running,
`"ollama": true`.

**Step 2 — open the UI**:

```bash
python3 -m http.server 5500 --directory docs   # then visit http://localhost:5500
```

Or just double-click `docs/index.html`. The ⚙ settings let you set the backend URL, result
count, toggle LLM rewriting, and toggle a grounded report. The header shows live status —
`sem on/off` (semantic embeddings) and `llm on/off` (Ollama).

---

## 5. Run the notebook

```bash
jupyter notebook ScoutRAG_CS455.ipynb
```

Run top to bottom. It cleans the data, runs EDA (figures → `eda_outputs/`), demonstrates the
search/ranking module, the grounding check, and the evaluation with baselines. The Colab copy
(`ScoutRAG_CS455_colab_backup.ipynb`) does the same with Colab-friendly paths.

---

## 6. Rebuild the embeddings manually (optional)

Only needed if you change `embed_text` (the play-style profile) or delete the cache and don't want
to wait for the backend to rebuild it lazily:

```bash
python build_embeddings.py
```

This encodes each player's play-style text and writes `player_embeddings.pkl` (raw-CSV-indexed,
the format the backend and notebook expect). Restart the backend afterward — the header should
read `sem on`.

---

## 7. How the engine works

```
query
  └─ intent routing (LLM or rule-based):  info / similar / search
  └─ (search) parse_query        → hard constraints (strict)  +  soft signals (ranking)
  └─ (optional) LLM fills a missing position/nationality/club & reinforces named soft attrs
  └─ structured_filter           → keep only rows satisfying every hard constraint
  └─ rank_players                → position-aware score from the 1–20 attributes only
  └─ (descriptive queries) semantic re-rank within the filtered pool
  └─ top-k   (or  "There is no player like that in our database.")
```

Key design choices, all aligned with the instructor's feedback:

- **Hard vs soft constraints.** Precise numbers (`age < 23`, `passing >= 15`, `value > €5M`) are
  enforced *strictly*; vague adjectives ("creative", "clinical") become *ranking signals*, upgraded
  to explicit attribute signals (e.g. creative → vision/passing) by the LLM when available.
- **The LLM can never invent or drop a constraint.** The original user query is the authoritative
  source of hard filters. The LLM rewrite may only fill a *missing* position/nationality/club and
  reinforce a soft signal for an attribute the user actually named — so "left-footed centre backs"
  never silently gains a pace constraint.
- **Ranking uses only the 1–20 player attributes** (position-appropriate), not age, nationality,
  name, or height — those matter only when the query explicitly asks for them. Player-name text is
  excluded from the embeddings so similarity reflects play-style, not names.
- **Constraint-aware similarity.** "Turkish players like Haaland" keeps only Turkish players, then
  ranks by embedding similarity to the reference.
- **Grounding verification** flags any player name or numeric stat in a generated report that is not
  present in the retrieved rows — the core contribution.

---

## 8. Evaluation, tests & ablations

Run **`ScoutRAG_Tests_and_Ablations.ipynb`** top to bottom for the instructor-requested
evaluation. It has three parts:

1. **Test queries** — a gallery of representative queries (multi-constraint search, OR
   conditions, numeric ranges, descriptive/soft terms, position-aware goalkeepers, the explicit
   no-match message, single-player info lookup, constraint-aware similarity, and a grounded
   report with the grounding check) so you can see exactly what each query returns.
2. **Quantitative metrics** over 40 scouting queries — constraint-satisfaction rate, result
   counts, and zero-result counts — comparing the baselines:
   structured-filter-only · vector-retrieval-only · **hybrid (ScoutRAG)**.
3. **Ablation studies** — (3.1) retrieval-strategy ablation, (3.2) semantic re-rank on vs off,
   (3.3) LLM rewrite on vs off, (3.4) parser robustness regression checks.

All tables/figures are saved to `eda_outputs/` (`eval_summary.csv`, `eval_per_query.csv`,
`ablation_semantic.csv`, `eval_constraint_satisfaction.png`) for direct use in the report.
Honest error analysis is written in the final notebook section — what failed is reported, not
hidden.

> Note: semantic-re-rank metrics and the LLM ablation only produce numbers on a machine with
> `sentence-transformers` (and Ollama running). Without them the notebook still runs and reports
> constraint satisfaction; the semantic/LLM rows are clearly marked as skipped.

---

## 9. Limitations (honest)

- FM24 ratings are designer-set game values, not validated real-world ability.
- "Strong / creative / clinical" thresholds are heuristic design choices, not learned.
- The rule-based parser can miss rare phrasings; the LLM rewrite mitigates but is best-effort.
- `value` / `wage` are missing for some lower-league or not-for-sale players, so budget filters
  exclude rows with unknown values by design.

---

## 10. Remaining work before the June 7 deadline

Code-side, the pipeline (cleaning → filtering → semantic retrieval → grounded generation →
grounding verification → evaluation) is implemented. What the **team** still needs to finish for
submission:

- [ ] **Final report (PDF, ~8–10 pages).** Problem, corrected framing, data + honesty note,
      method, evaluation tables, baseline comparison, **honest error analysis**, limitations,
      future work. Reuse the figures in `eda_outputs/`.
- [ ] **Run the full evaluation on a machine with Ollama + sentence-transformers** and freeze the
      final numbers (the sandbox can only use the attribute-vector fallback). Paste the real tables
      into the report.
- [ ] **Demo materials** — short walkthrough / screen recording of a few representative queries
      (a hard-constraint search, a descriptive search, a "similar to X" query, and a grounded
      report with the verification step).
- [ ] **LLM-usage disclosure** in the report — which parts of code/writing used an AI assistant
      (required by the academic-integrity section).
- [ ] **Code zip** for SUCourse: clean repo + this README (exclude `.venv/`, `__pycache__/`,
      `.idea/`, and the large `player_embeddings.pkl`).
- [ ] **Confirm the corrected scope in the report**: standard CS455 track, *not* Paper Track and
      *not* Text-to-SQL — match the framing in this README.

Division of labor (from the proposal): Umut Can (data + retrieval engine), Yiğit Onur
(database/structured filtering), Mustafa Ege (LLM + evaluation/report). Adjust as needed and state
the final split in the report.

---

## 11. Deployment — local only

This project runs **entirely on your own machine** — there is no hosted/online deployment. The
engine is a Python (FastAPI) server, so it cannot run on a static host. To run it, get the code
(clone the repo, **or** unzip the submitted `.zip`), then run these commands from the project
folder:

```bash
# 1. set up once (skip if already done — see §3)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (optional) start the local LLM in a separate terminal
ollama serve                       # needs: brew install ollama && ollama pull llama3.1

# 3. start the backend (first run builds the embedding cache once)
uvicorn backend.app:app --reload --port 8000

# 4. in another terminal, serve the UI and open it
python3 -m http.server 5500 --directory docs   # then visit http://localhost:5500
```

That's the whole demo: backend on `http://localhost:8000`, UI on `http://localhost:5500`, fully
offline. Double-clicking `docs/index.html` also works; just make sure the backend is running and
the ⚙ Backend URL is `http://localhost:8000`.

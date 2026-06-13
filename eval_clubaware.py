"""
Club-aware re-evaluation of ScoutRAG  (does NOT modify any engine/system code).

Only difference vs scoutrag_eval.py:
    parse_query(...) is called with known_clubs=<all clubs in the dataset> — exactly
    what the live app (backend/app.py, app_gradio.py) does. The original harness omitted
    known_clubs, so every "at <club>" constraint was silently dropped, producing phantom
    failures and understating recall.

SEMANTIC: ON automatically if `sentence-transformers` is importable (same as the original
          harness). Run this where it is installed (your machine / Colab) for the official
          semantic-ON numbers. pass/fail (Recall@20) is INVARIANT to semantic on/off —
          semantic only re-orders players already inside the filtered pool; it never adds
          or removes a candidate. (Verified: with semantic OFF this script reproduces the
          original numbers exactly — 71/95, ablation struct 30/39, semantic-only 33.3%.)
          So you get the same 77.9% / 84.6% either way; only per-row top5 ordering differs.

LLM:      the evaluation NEVER calls the LLM (no rewrite/route/generate in the harness),
          so Ollama on/off cannot change these numbers. LLM only affects the live app.

Run:
    cd <project root>           # where fmdata24llm.csv and player_embeddings.pkl live
    python eval_clubaware.py    # writes eval_results.csv and ablation.csv
"""
import warnings, csv, ast, unicodedata, time
warnings.filterwarnings("ignore")
from scoutrag.core import (load_and_clean, build_profiles, align_embeddings,
                           parse_query, search, find_players_by_name, players_to_records,
                           rank_players)

TESTS = [ast.literal_eval(n.value) for n in ast.parse(open("scoutrag_eval.py").read()).body
         if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "TESTS" for t in n.targets)][0]

df = build_profiles(load_and_clean("fmdata24llm.csv", verbose=False))
emb = align_embeddings(df, "player_embeddings.pkl")
clubs = sorted(df["Club_Name"].dropna().unique().tolist())

embed_query_fn = None
try:
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer("all-MiniLM-L6-v2")
    embed_query_fn = lambda t: _m.encode([t], normalize_embeddings=True)[0].astype("float32")
    print("semantic search: ENABLED")
except Exception as e:
    print(f"semantic search: DISABLED ({e}) -- pass/fail identical, only ordering differs")
SEM_ON = emb is not None and embed_query_fn is not None

def _s(s):
    s = unicodedata.normalize("NFKD", str(s)); s = "".join(c for c in s if not unicodedata.combining(c))
    return (s.replace("ı","i").replace("İ","i").replace("ş","s").replace("ğ","g")
             .replace("ü","u").replace("ö","o").replace("ç","c").lower().strip())
def inr(recs, t): t=_s(t); return any(t in _s(p.get("Player_Name","")) for p in recs)
def trace(p): return " | ".join(p.get("trace", [])) or "(none)"

def run_search(q):
    p = parse_query(q, known_clubs=clubs); p["top_k"] = 20
    return search(df, p, embeddings=emb, embed_query_fn=embed_query_fn, return_df=True), p
def run_struct(q):
    p = parse_query(q, known_clubs=clubs); p["top_k"] = 20
    return search(df, p, embeddings=None, embed_query_fn=None, return_df=True), p
def run_sem(q):
    if not SEM_ON: return None
    p = parse_query(q, known_clubs=clubs); p["top_k"] = 20
    qv = embed_query_fn(q); scores = (emb @ qv).astype(float)
    return players_to_records(rank_players(df, p, sem_scores=scores).head(20))

RESULT_COLS = ["id","category","query","target_player","note","expected_found","expected_empty",
    "n_results","top5_players","parsed_tokens","target_in_results","results_empty","pass","fail_reason","latency_s"]
rows = []
for t in TESTS:
    st=time.perf_counter(); ttype=t["test_type"]; tg=t.get("target_player"); q=t["query"]
    ef=t["expected_found"]; ee=t["expected_empty"]; cat=t["category"]
    if ttype=="lookup":
        recs=players_to_records(find_players_by_name(df,q,limit=20)); n=len(recs)
        tir=inr(recs,tg) if tg else False; empty=n==0; tok="(name lookup)"
    else:
        res,p=run_search(q); recs=res.get("players",[]); n=len(recs)
        tir=inr(recs,tg) if tg else False; empty=n==0 or bool(res.get("message")); tok=trace(p)
    passed=True; fr=""
    if ee:
        if not empty: passed=False; fr=f"expected empty but got {n} results"
    else:
        if empty and not (cat=="edge_case" and not tg): passed=False; fr="expected results but got none"
    if ef and not tir: passed=False; fr=(fr+f" | target '{tg}' not in top-20") if fr else f"target '{tg}' not in top-20"
    if (not ef) and tir and tg: passed=False; fr=f"target '{tg}' appeared but should NOT have"
    rows.append({"id":t["id"],"category":cat,"query":q,"target_player":tg or "","note":t["note"],
        "expected_found":ef,"expected_empty":ee,"n_results":n,
        "top5_players":"; ".join(r.get("Player_Name","?") for r in recs[:5]),"parsed_tokens":tok,
        "target_in_results":tir,"results_empty":empty,"pass":passed,"fail_reason":fr,
        "latency_s":round(time.perf_counter()-st,3)})
npass=sum(r["pass"] for r in rows)
with open("eval_results.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=RESULT_COLS); w.writeheader(); w.writerows(rows)
print(f"eval_results.csv : {npass}/{len(rows)} = {npass/len(rows)*100:.1f}% Recall@20")

ABL={"real_player_lookup","gk_attributes","attribute_coverage"}
cat1=[t for t in TESTS if t["category"] in ABL and t.get("target_player")]
ABL_COLS=["id","query","target_player","struct_found","struct_n","semantic_found","semantic_n","hybrid_found","hybrid_n","winner"]
arows=[]
for t in cat1:
    tg=t["target_player"]; q=t["query"]
    rs,_=run_struct(q); sf=inr(rs.get("players",[]),tg); sn=len(rs.get("players",[]))
    rh,_=run_search(q); hf=inr(rh.get("players",[]),tg); hn=len(rh.get("players",[]))
    if SEM_ON: sem=run_sem(q); semf=inr(sem,tg); semn=len(sem)
    else: semf,semn=False,0
    modes=[m for m,b in [("structured",sf),("semantic",semf),("hybrid",hf)] if b]
    arows.append({"id":t["id"],"query":q,"target_player":tg,"struct_found":sf,"struct_n":sn,
        "semantic_found":semf,"semantic_n":semn,"hybrid_found":hf,"hybrid_n":hn,
        "winner":"+".join(modes) if modes else "none"})
with open("ablation.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=ABL_COLS); w.writeheader(); w.writerows(arows)
na=len(arows)
print(f"ablation.csv     : struct {sum(r['struct_found'] for r in arows)}/{na} | "
      f"semantic {sum(r['semantic_found'] for r in arows)}/{na} | hybrid {sum(r['hybrid_found'] for r in arows)}/{na}")

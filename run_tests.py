"""Run all 40 ScoutRAG test prompts against the local backend and save results."""

import json
import csv
import time
import urllib.request
import urllib.error

API = "http://localhost:8000"

TESTS = [
    # (id, prompt, use_llm, make_report, category)
    # Cat 1: Single Player Info Lookup
    (1,  "Lionel Messi",                          True,  False, "Player Info"),
    (2,  "Tell me about Erling Haaland",           True,  False, "Player Info"),
    (3,  "Who is Luka Modrić",                     True,  False, "Player Info"),
    (4,  "Kylian Mbappe",                          True,  False, "Player Info"),

    # Cat 2: Player Not Found
    (5,  "Tell me about Ege Özer",                 True,  False, "Not Found"),
    (6,  "Who is John Xyzzyspoon",                 True,  False, "Not Found"),
    (7,  "asdfghjkl",                              True,  False, "Not Found"),

    # Cat 3: Similarity
    (8,  "5 players similar to Kylian Mbappe",     True,  False, "Similarity"),
    (9,  "Players like Lionel Messi",              True,  False, "Similarity"),
    (10, "Turkish players similar to Haaland",     True,  False, "Similarity"),

    # Cat 4: Basic Single-Constraint
    (11, "Striker",                                True,  False, "Single Constraint"),
    (12, "Brazilian midfielder",                   True,  False, "Single Constraint"),
    (13, "Left-footed centre-back",                True,  False, "Single Constraint"),
    (14, "Players under 21",                       True,  False, "Single Constraint"),
    (15, "Goalkeeper with reflexes >= 17",          True,  False, "Single Constraint"),

    # Cat 5: Multi-Constraint
    (16, "Left-footed centre-back under 23, passing > 15, valued under €10M",
                                                   True,  False, "Multi-Constraint"),
    (17, "Top 5 strikers with finishing >= 16 and pace >= 15",
                                                   True,  False, "Multi-Constraint"),
    (18, "German goalkeeper, handling >= 15, kicking >= 14",
                                                   True,  False, "Multi-Constraint"),
    (19, "Right-footed winger under 25, dribbling >= 16, pace >= 15, valued under €20M",
                                                   True,  False, "Multi-Constraint"),
    (20, "Defensive midfielder over 28, tackling >= 15, passing >= 14, stamina >= 15",
                                                   True,  False, "Multi-Constraint"),

    # Cat 6: Descriptive / Soft Terms
    (21, "Clinical striker",                       True,  False, "Descriptive"),
    (22, "Creative playmaker with great vision",   True,  False, "Descriptive"),
    (23, "Fast and strong winger",                 True,  False, "Descriptive"),
    (24, "Commanding goalkeeper with good reflexes", True, False, "Descriptive"),
    (25, "Tireless box-to-box midfielder",         True,  False, "Descriptive"),
    (26, "A brave and aggressive centre-back who is good in the air",
                                                   True,  False, "Descriptive"),

    # Cat 7: OR Conditions & Special Filters
    (27, "Turkish or Brazilian winger under 21",   True,  False, "OR / Special"),
    (28, "Left-back who plays for Beşiktaş",       True,  False, "OR / Special"),
    (29, "Players aged between 18 and 20",         True,  False, "OR / Special"),
    (30, "Top 3 wonderkid strikers",               True,  False, "OR / Special"),

    # Cat 8: Edge Cases
    (31, "Striker with finishing >= 20",            True,  False, "Edge Case"),
    (32, "Left-footed Nigerian goalkeeper under 19 with reflexes >= 18",
                                                   True,  False, "Edge Case"),
    (33, "Give me 50 players",                     True,  False, "Edge Case"),
    (34, "   ",                                    True,  False, "Edge Case"),
    (35, "Find me a fast, creative, clinical, strong, tireless striker under 21 valued under €5M",
                                                   True,  False, "Edge Case"),
    (36, "Messi Ronaldo",                          True,  False, "Edge Case"),

    # Cat 9: Grounded Report (report ON)
    (37, "Top 5 clinical strikers under 25",       True,  True,  "Grounded Report"),
    (38, "Best 3 creative midfielders",            True,  True,  "Grounded Report"),

    # Cat 10: LLM Off
    (39, "A clinical, visionary number 10 who is cheap",
                                                   False, False, "LLM Off"),
    (40, "Tell me about Phil Foden",               False, False, "LLM Off"),
]


def call_api(prompt, use_llm, make_report, timeout=120):
    """Send a search request to the backend and return the JSON response."""
    body = json.dumps({
        "query": prompt,
        "top_k": 10,
        "use_llm": use_llm,
        "make_report": make_report,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode("utf-8", "ignore")}
    except Exception as e:
        return {"error": str(e)}


def summarize(result):
    """Extract a short summary dict from the API response."""
    if "error" in result:
        return {
            "mode": "ERROR",
            "num_players": 0,
            "top_player": "",
            "constraints": "",
            "summary_text": result.get("error", ""),
            "rewritten": "",
            "used_llm": False,
            "grounding_score": "",
            "report_excerpt": "",
        }

    players = result.get("players", [])
    top = players[0]["Player_Name"] if players else ""
    top3 = ", ".join(p["Player_Name"] for p in players[:3]) if players else ""

    gr = result.get("grounding", {})
    gr_score = f'{gr["grounding_score"]*100:.0f}%' if gr and "grounding_score" in gr else ""

    report = result.get("report", "")
    report_excerpt = (report[:150] + "...") if report and len(report) > 150 else report

    return {
        "mode": result.get("mode", "?"),
        "num_players": len(players),
        "top_player": top,
        "top_3": top3,
        "constraints": " | ".join(result.get("constraints", [])),
        "summary_text": (result.get("summary") or result.get("message") or "")[:300],
        "rewritten": result.get("rewritten_query") or "",
        "used_llm": result.get("used_llm", False),
        "grounding_score": gr_score,
        "report_excerpt": report_excerpt or "",
    }


import sys

def main():
    # Force stdout to use UTF-8 to prevent encoding issues with special characters in player names
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    print("=" * 70)
    print("  ScoutRAG Test Runner — 40 Test Cases")
    print("=" * 70)

    # Check backend is up
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        print(f"  Backend: OK | Players: {health['players']} | "
              f"Semantic: {health['semantic']} | Ollama: {health['ollama']}")
    except Exception as e:
        print(f"  ERROR: Cannot reach backend at {API} — {e}")
        print("  Start the backend first: uvicorn backend.app:app --reload --port 8000")
        return

    print("=" * 70)

    all_results = []
    csv_rows = []

    for tid, prompt, use_llm, make_report, category in TESTS:
        display = prompt.strip() if prompt.strip() else "(empty/whitespace)"
        print(f"\n[{tid:02d}/{len(TESTS)}] {category}")
        print(f"  Prompt:    {display}")
        print(f"  LLM={use_llm}  Report={make_report}")

        t0 = time.time()
        result = call_api(prompt, use_llm, make_report)
        elapsed = time.time() - t0

        s = summarize(result)
        print(f"  Mode:      {s['mode']}")
        print(f"  Players:   {s['num_players']}")
        if s["top_3"]:
            print(f"  Top 3:     {s['top_3']}")
        if s["rewritten"]:
            print(f"  Rewritten: {s['rewritten']}")
        if s["grounding_score"]:
            print(f"  Grounding: {s['grounding_score']}")
        summary_preview = s["summary_text"][:120]
        if summary_preview:
            print(f"  Summary:   {summary_preview}...")
        print(f"  Time:      {elapsed:.1f}s")

        # Store full result
        all_results.append({
            "test_id": tid,
            "category": category,
            "prompt": prompt,
            "use_llm": use_llm,
            "make_report": make_report,
            "elapsed_s": round(elapsed, 2),
            "response": result,
        })

        # CSV row
        csv_rows.append({
            "test_id": tid,
            "category": category,
            "prompt": prompt,
            "use_llm": use_llm,
            "make_report": make_report,
            "mode": s["mode"],
            "num_players": s["num_players"],
            "top_player": s["top_player"],
            "top_3": s["top_3"],
            "constraints": s["constraints"],
            "summary": s["summary_text"],
            "rewritten_query": s["rewritten"],
            "used_llm": s["used_llm"],
            "grounding_score": s["grounding_score"],
            "report_excerpt": s["report_excerpt"],
            "elapsed_s": round(elapsed, 2),
        })

    # Save JSON (full data)
    json_path = "test_results_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n{'=' * 70}")
    print(f"  Full results saved to:  {json_path}")

    # Save CSV (summary table)
    csv_path = "test_results_summary.csv"
    fields = list(csv_rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  Summary CSV saved to:   {csv_path}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    total = len(TESTS)
    errors = sum(1 for r in all_results if "error" in r["response"])
    empty = sum(1 for r in all_results if len(r["response"].get("players", [])) == 0
                and "error" not in r["response"])
    with_players = total - errors - empty
    print(f"  Total tests:      {total}")
    print(f"  Returned players: {with_players}")
    print(f"  Empty results:    {empty}")
    print(f"  Errors:           {errors}")
    avg_time = sum(r["elapsed_s"] for r in all_results) / total
    print(f"  Avg response:     {avg_time:.1f}s")
    print(f"{'=' * 70}")
    print("  Done!")


if __name__ == "__main__":
    main()

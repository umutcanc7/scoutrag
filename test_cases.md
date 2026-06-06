# ScoutRAG — 40 Test Prompts

> **How to use:** Enter each prompt into the website's chat box. Record the output (screenshot or copy-paste). Compare the result against the "What to check" criteria.
>
> **Settings:** Run tests 1–36 with **LLM on**, **Report off**. Tests 37–38 need **Report on**. Tests 39–40 need **LLM off**.

---

## Category 1 · Single Player Info Lookup (existing players)

These test the `info` intent — the system should find the player and show their profile card + LLM summary.

| # | Prompt | What to check |
|---|--------|---------------|
| 1 | `Lionel Messi` | Returns Messi's profile card with correct nationality (Argentinian), position, club, attributes. Summary should describe him. |
| 2 | `Tell me about Erling Haaland` | Tests the "tell me about" prefix — should route to `info` intent, show Haaland's card. |
| 3 | `Who is Luka Modrić` | Tests the "who is" prefix + accented character (ć). Should find Modrić and return his profile. |
| 4 | `Kylian Mbappe` | Bare name lookup without accents (Mbappé). Tests accent-insensitive matching. Should find him. |

---

## Category 2 · Player Not Found

These test the new LLM-powered "not found" response. The message should be concise, professional, and NOT overly apologetic.

| # | Prompt | What to check |
|---|--------|---------------|
| 5 | `Tell me about Ege Özer` | Player not in DB. Should return an empty player list, no "Player profile:" label, and a short LLM-generated message (not the old hardcoded string). |
| 6 | `Who is John Xyzzyspoon` | Completely fictional name. Should get a clean not-found message, no crash. |
| 7 | `asdfghjkl` | Gibberish input. Should handle gracefully — either not-found or empty search results. No crash. |

---

## Category 3 · Similarity Queries

These test the `similar` intent — finding players with a similar play-style to a reference player.

| # | Prompt | What to check |
|---|--------|---------------|
| 8 | `5 players similar to Kylian Mbappe` | Should return 5 players ranked by similarity. Mode should be `similar`. Players should be outfielders (not GKs). |
| 9 | `Players like Lionel Messi` | Should return similar players (default count). Check that Messi himself is NOT in the results. |
| 10 | `Turkish players similar to Haaland` | **Constraint-aware similarity.** Should return players similar to Haaland who are also Turkish. Verify every result has Turkish nationality. |

---

## Category 4 · Basic Single-Constraint Search

Each prompt tests one constraint type in isolation.

| # | Prompt | What to check |
|---|--------|---------------|
| 11 | `Striker` | Position filter only. All results should have ST in their position. |
| 12 | `Brazilian midfielder` | Position + nationality. All results should be Brazilian and have M in position. |
| 13 | `Left-footed centre-back` | Foot + position. All results should be left-footed and play as D(C)/D(LC)/D(RLC). |
| 14 | `Players under 21` | Age filter. Every player's age should be ≤ 20. |
| 15 | `Goalkeeper with reflexes >= 17` | Position (GK) + hard attribute constraint. All results should be GK with Ref ≥ 17. |

---

## Category 5 · Multi-Constraint Search

Complex queries combining multiple filter types. These are the core scouting use case.

| # | Prompt | What to check |
|---|--------|---------------|
| 16 | `Left-footed centre-back under 23, passing > 15, valued under €10M` | Foot=Left, position=CB, age ≤ 22, Pas > 15, Value ≤ €10M. Verify ALL constraints on every result. |
| 17 | `Top 5 strikers with finishing >= 16 and pace >= 15` | top_k=5, position=ST, Fin ≥ 16, Pac ≥ 15. Should return exactly 5 results (or fewer if not enough match). |
| 18 | `German goalkeeper, handling >= 15, kicking >= 14` | Nationality=German, position=GK, Han ≥ 15, Kic ≥ 14. |
| 19 | `Right-footed winger under 25, dribbling >= 16, pace >= 15, valued under €20M` | 4 hard constraints + foot + position + age + value. The toughest filter combo. |
| 20 | `Defensive midfielder over 28, tackling >= 15, passing >= 14, stamina >= 15` | Position=DM, age ≥ 29, Tck ≥ 15, Pas ≥ 14, Sta ≥ 15. Tests veteran + multiple attrs. |

---

## Category 6 · Descriptive / Soft Term Search

These use scouting adjectives instead of numbers. The engine should map them to soft ranking signals.

| # | Prompt | What to check |
|---|--------|---------------|
| 21 | `Clinical striker` | Should rank by finishing (Fin). Top results should have high Fin values. |
| 22 | `Creative playmaker with great vision` | Should boost vision (Vis) and possibly passing. Top results should have high Vis/Pas. |
| 23 | `Fast and strong winger` | Should boost pace (Pac) and strength (Str). Top results should have high values in both. |
| 24 | `Commanding goalkeeper with good reflexes` | GK-specific soft terms. Should boost command of area (Cmd) and reflexes (Ref). |
| 25 | `Tireless box-to-box midfielder` | Should boost stamina (Sta) and work rate (Wor). Position should be M(C). |
| 26 | `A brave and aggressive centre-back who is good in the air` | Should boost bravery (Bra), aggression (Agg), heading (Hea). |

---

## Category 7 · OR Conditions & Special Filters

| # | Prompt | What to check |
|---|--------|---------------|
| 27 | `Turkish or Brazilian winger under 21` | OR nationality (Turkish OR Brazilian). All results should be one of the two nationalities AND a winger AND age ≤ 20. |
| 28 | `Left-back who plays for Beşiktaş` | Club filter with Turkish characters. Should find left-backs at Beşiktaş. Accent-insensitive matching. |
| 29 | `Players aged between 18 and 20` | Age range filter. All results should have age 18, 19, or 20. |
| 30 | `Top 3 wonderkid strikers` | Tests "wonderkid" synonym (age ≤ 20) + position + top_k=3. Should return 3 young strikers. |

---

## Category 8 · Edge Cases & Stress Tests

| # | Prompt | What to check |
|---|--------|---------------|
| 31 | `Striker with finishing >= 20` | Extreme attribute threshold. Very few (or zero) players should have Fin=20. Might return "no match" or a tiny list. |
| 32 | `Left-footed Nigerian goalkeeper under 19 with reflexes >= 18` | Very narrow filter — likely 0 results. Should show a clean "no player found" message, not crash. |
| 33 | `Give me 50 players` | Tests large top_k. Should return up to 50 players ranked by overall quality (no constraints). |
| 34 | `           ` | Empty/whitespace input. Should handle gracefully — either ignore or show an error message. No crash. |
| 35 | `Find me a fast, creative, clinical, strong, tireless striker under 21 valued under €5M` | Extreme multi-soft + multi-hard. Tests if the system can handle many signals at once without breaking. Likely very few results. |
| 36 | `Messi Ronaldo` | Two player names at once. Unclear intent. Observe how the system handles ambiguity — does it find one, both, or treat it as a search? |

---

## Category 9 · Grounded Report (toggle Report ON in settings ⚙)

> [!IMPORTANT]
> Turn on **"Grounded report"** in the ⚙ settings panel before running these two.

| # | Prompt | What to check |
|---|--------|---------------|
| 37 | `Top 5 clinical strikers under 25` | Should return players + a written scouting report below them. Check grounding score (should be high, ideally 100%). Verify that every name and stat mentioned in the report actually appears in the result cards. |
| 38 | `Best 3 creative midfielders` | Same check — report text should only reference players and stats from the retrieved data. Grounding score visible. |

---

## Category 10 · LLM Off (toggle LLM OFF in settings ⚙)

> [!IMPORTANT]
> Turn **off** "LLM query rewriting" in the ⚙ settings panel before running these two. This tests the rule-based fallback.

| # | Prompt | What to check |
|---|--------|---------------|
| 39 | `A clinical, visionary number 10 who is cheap` | Without LLM, the rule-based parser should still map "clinical" → Fin, "visionary" → Vis, "number 10" → ST(C)/AM(C), "cheap" might not parse. Compare results to the same query with LLM on. |
| 40 | `Tell me about Phil Foden` | Info lookup without LLM. The heuristic router should still detect the "Tell me about" pattern and return Foden's profile card. |

---

## Summary Checklist

| Category | Tests | What it validates |
|----------|-------|-------------------|
| Player info lookup | 1–4 | Name matching, accent handling, intent routing |
| Player not found | 5–7 | LLM-generated apology, graceful error handling |
| Similarity | 8–10 | Embedding similarity, constraint-aware similarity |
| Single constraint | 11–15 | Position, nationality, foot, age, GK attributes |
| Multi-constraint | 16–20 | Strict filtering with multiple simultaneous constraints |
| Descriptive terms | 21–26 | Soft adjective → attribute mapping, ranking signals |
| OR / special | 27–30 | OR nationality, club matching, age ranges, synonyms |
| Edge cases | 31–36 | Extreme values, empty input, ambiguity, stress |
| Grounded report | 37–38 | Report generation + grounding verification score |
| LLM off fallback | 39–40 | Rule-based parser + heuristic router without Ollama |

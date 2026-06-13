# Data Source Documentation
## CS455 §8 Compliance Statement

### Dataset
**Football Manager 2024 (FM24) in-game player export**

### How the data was obtained

Football Manager 2024 includes a **built-in, officially supported scouting-view export feature**.
Within the game's squad/scouting screen, a user can select visible columns and trigger an export
command that writes the current view to an **RTF file** on the local machine. No third-party tools,
no web scraping, no reverse engineering, and no network requests to external servers were involved.

Steps we followed:
1. Opened the global scouting view in Football Manager 2024.
2. Configured the visible columns to include all relevant player attributes (technical, mental, physical, and goalkeeper-specific FM24 stats on the 1–20 scale, plus metadata such as name, age, nationality, club, position, transfer value, and wage).
3. Used the game's built-in **"Export to RTF"** command (available in the game's view menu).
4. Converted the resulting RTF file to CSV using `convert_fm_table.py` (included in the repository).

### Terms of Service

The export is an **official, documented in-game feature** provided by Sports Interactive / SEGA.
Using a licensed copy of the game to export data to a local file for personal/academic research
does not violate the Football Manager End User Licence Agreement. No data was scraped from
Sports Interactive servers, the Steam platform, or any third-party website.

### Data nature & privacy

All player attributes in the CSV are **synthetic, designer-set game values** — they represent
fictional game statistics, not biographical or sensitive personal data of real individuals. Player
names are used as game identifiers; no private, medical, or financial data of real persons is
included. Therefore, no data anonymization or data-protection obligation (e.g., GDPR) applies.

### Limitations (honest)

Because the values are set by game designers rather than measured from real performance, the
dataset is used strictly as a **prototype validation environment** for the retrieval and grounding
pipeline. We do **not** claim FM24 attributes are accurate representations of real-world player
ability, and the system is framed accordingly throughout the report.

### Files

| File | Description |
|---|---|
| `fmdata24llm.csv` | Processed dataset (~42,500 rows after cleaning, 57 attribute columns + metadata) |
| `convert_fm_table.py` | Script that converts the raw FM24 RTF export to a clean CSV |
| `scoutrag/core.py` → `load_and_clean()` | Further cleaning and normalization applied at load time |

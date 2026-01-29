import json
from pathlib import Path
import pandas as pd

pops_path = Path("docs/data/populations.json")
xlsx_path = Path("data/anc_pop.xlsx")

# Read existing populations.json (contains wards)
with pops_path.open("r", encoding="utf-8-sig") as f:
    payload = json.load(f)

df = pd.read_excel(xlsx_path, sheet_name=0)

# Try to find columns that look like ANC id + population
# We'll be forgiving because sheet headers vary.
cols = [c for c in df.columns]
anc_col = cols[0]
pop_col = cols[1] if len(cols) > 1 else cols[0]

df = df.rename(columns={anc_col: "ANC", pop_col: "POP"}).copy()
df["ANC"] = df["ANC"].astype(str).str.strip()

# Keep rows that look like "1A", "2B", etc.
df = df[df["ANC"].str.match(r"^\d[A-Z]$")].copy()

df["POP"] = pd.to_numeric(df["POP"], errors="coerce")

ancs = {}
for _, r in df.iterrows():
    if pd.isna(r["POP"]):
        continue
    ancs[str(r["ANC"])] = int(r["POP"])

payload["ancs"] = ancs

pops_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote {pops_path} with {len(ancs)} ANC populations")

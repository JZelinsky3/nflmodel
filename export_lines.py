import pandas as pd, nfl_data_py as ndp

years = list(range(2018, 2026))
print("fetching betting lines for", years)
df = ndp.import_betting_lines(years)

# normalize to the column names your model expects
rename_map = {
    "spread_close": "spread_line_close",
    "spread_line": "spread_line_close",
    "total_close": "total_line_close",
    "total_line": "total_line_close",
    "home_moneyline": "home_moneyline_close",
    "away_moneyline": "away_moneyline_close",
}
for src, dst in rename_map.items():
    if src in df.columns and dst not in df.columns:
        df[dst] = df[src]

needed = ["game_id","spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]
for c in needed:
    if c not in df.columns:
        df[c] = pd.NA

out = df[["game_id"] + needed[1:]].drop_duplicates("game_id")
out.to_csv("betting_lines_2018_2025.csv", index=False)
print("wrote betting_lines_2018_2025.csv", out.shape)

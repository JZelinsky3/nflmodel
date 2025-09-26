import pandas as pd
from pathlib import Path

MASTER = "betting_lines_2018_2025.csv"
WEEK_FILE = "week4_lines_template.csv"   # after you filled it

def normalize_columns(df):
    # Ensure the 5 expected columns exist
    for c in ["game_id","spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df[["game_id","spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]]

def main():
    wk = pd.read_csv(WEEK_FILE)
    wk = normalize_columns(wk)

    if Path(MASTER).exists():
        master = pd.read_csv(MASTER)
        master = normalize_columns(master)
        # overwrite rows that appear in wk, keep others
        not_overwritten = master[~master["game_id"].isin(wk["game_id"])]
        merged = pd.concat([not_overwritten, wk], ignore_index=True)
    else:
        merged = wk

    merged = merged.drop_duplicates("game_id")
    merged.to_csv(MASTER, index=False)
    print("Updated", MASTER, "rows:", len(merged))

if __name__ == "__main__":
    main()

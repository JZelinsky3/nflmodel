import polars as pl

try:
    import nflreadpy as nfl
except Exception as e:
    raise SystemExit(f"nflreadpy not installed: {e!r}")

YEARS = list(range(2018, 2026))
print("loading pbp for", YEARS)
pbp = nfl.load_pbp(YEARS)  # polars DataFrame in your setup

# Ensure columns exist
for c in ["game_id", "spread_line", "total_line"]:
    if c not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(None).alias(c))

# One row per game_id with closing spread/total
out = (
    pbp
    .select(["game_id", "spread_line", "total_line"])
    .unique(subset=["game_id"])
    .rename({
        "spread_line": "spread_line_close",
        "total_line": "total_line_close",
    })
    .with_columns([
        pl.lit(None).alias("home_moneyline_close"),
        pl.lit(None).alias("away_moneyline_close"),
    ])
    .select([
        "game_id",
        "spread_line_close",
        "total_line_close",
        "home_moneyline_close",
        "away_moneyline_close",
    ])
)

out.write_csv("betting_lines_2018_2025.csv")
print("wrote betting_lines_2018_2025.csv", out.shape)

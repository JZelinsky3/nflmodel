## Historical data file

This repo excludes the large historical CSV from git.  
Download it from the Releases page and place it in the project root.

Steps
1. Open the repo on GitHub
2. Click Releases
3. Open the latest release named Historical data pack
4. Download the asset historical_with_features.csv
5. Move the file into the project root folder next to index.html and pick_board.py

Optional verify checksum
If the release lists a SHA256 hash you can verify it
macOS or Linux
  shasum -a 256 historical_with_features.csv
Windows PowerShell
  Get-FileHash historical_with_features.csv -Algorithm SHA256
Compare with the value shown on the release page

## What to do after cloning

Clone or download the repo code.

git clone https://github.com/JZelinsky3/nflmodel.git
cd nflmodel

Go to the GitHub Releases page for the repo and download the big CSV.
Save it into the same folder you cloned into. The file should sit at
nflmodel slash historical_with_features csv

Create the environments from the requirements files

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 -m venv .linesenv
source .linesenv/bin/activate
pip install -r requirements-lines.txt
deactivate

Run the pipeline

source .venv/bin/activate
python model_with_features.py
python pick_board.py

Open the dashboard
Open index html with VS Code Live Server
Or open it directly in a browser if your CSV fetch paths are relative

## Common questions

Where do I put the CSV
Place it at the repo root next to index html. Your code already looks for historical_with_features csv in that location.

Can I rename the file
Do not rename it unless you also update any code that reads it. Keep the name consistent.

Can I ship a small sample for preview
Yes. Include historical_with_features_sample csv in the repo for demo. In the README say that the full dataset becomes available after downloading the release asset.

What if I update the CSV later
Create a new release. Use a new tag like v data two. Upload the new csv. Update the README to say Use the latest release.

What if the release asset is larger than one gigabyte
Your file is 136 MB which is fine. GitHub release assets can be much larger than a normal git blob, so you are safe here.

## What data the Model uses

Here is exactly what your current model uses, straight from the code you shared.

Market baseline that serves as the prior

Primary source is the closing spread. It converts spread_line_close into a home win probability with a logistic curve using K_SPREAD = 0.13.

Fallback is the moneylines. It converts home_moneyline_close and away_moneyline_close to implied win chances, then normalizes so home plus away equals one.

If neither spread nor moneylines are present, it uses 0.50 as a neutral prior.

That baseline is used as the offset in a binomial GLM with logit link. So the market sets the starting point and the model only learns corrections.

Residual features that supply the corrections
All features are built from team level rolling stats, using last three games only, shifted so the current game is not leaked.

Rest difference
rest_diff = home_rest_days minus away_rest_days where rest days are computed from each team’s previous game date.

Form difference using margin of victory
form3_diff = home_form3_mov minus away_form3_mov where form3_mov is the 3 game rolling mean of signed point margin for each team.

Offensive efficiency difference by EPA
off_epa_diff = home_off_epa_roll minus away_off_epa_roll where off_epa_roll is the 3 game rolling mean of a team’s per play offensive EPA from play by play.

Defensive efficiency difference by EPA allowed
def_epa_diff = home_def_epa_roll minus away_def_epa_roll where def_epa_allowed_game is negated first so that lower allowed EPA corresponds to better defense, then a 3 game rolling mean is taken.

Engineering details that affect predictions

Rolling windows are shifted by one game, so only past games feed each feature.

Missing rolling values at the start of a season are filled to sane defaults
rest days 7, form 0, rolling offensive EPA 0, rolling defensive EPA 0.

Features are standardized with StandardScaler before fitting.

Any feature that is constant or NaN across the training rows is dropped automatically.

Optional isotonic calibration is trained on finished games to make predicted probabilities well calibrated. If present, the calibrated column is used for upcoming outputs.

What does not feed the model right now

Totals are loaded but not used as a predictor.

Injuries and depth chart info are not merged in yet.

Any EPA beyond the simple per play averages is not used yet such as success rate or drive level EPA.

So the prediction for home win comes from
market implied home win chance as an offset
plus a learned correction from four inputs
rest diff, form diff, offensive EPA diff, defensive EPA diff.

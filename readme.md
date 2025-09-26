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

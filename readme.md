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

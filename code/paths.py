"""Folder layout of the project ():
  code/          these scripts
  data_public/   public data (download_public_data.sh) and fomc_meetings.csv
  results/       outputs
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
DATA = os.path.join(ROOT, "data_public")
FRENCH = os.path.join(DATA, "french")
FED = os.path.join(DATA, "fed")
MEETINGS = os.path.join(DATA, "fomc_meetings.csv")
RESULTS = os.path.join(ROOT, "results")

from pathlib import Path
from motion_heatmap import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subject", choices=["ASD", "TD"])
args = parser.parse_args()

PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data/raw/", args.subject)

file_lists = []
for folder in DATA_PATH.glob("*"):
    for file in folder.glob("*high*.MP4"):
        file_lists.append(file)

for file in file_lists:
    motion_heatmap_code(file)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.csv_loader import load_csv


csv_path = "data/raw/project_tasks.csv"

text = load_csv(csv_path)

print("----- EXTRACTED CSV TEXT -----")
print(text)
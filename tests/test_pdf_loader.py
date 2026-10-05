import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.pdf_loader import load_pdf


pdf_path = "data/raw/test_project.pdf"

text = load_pdf(pdf_path)

print("----- EXTRACTED TEXT -----")
print(text)
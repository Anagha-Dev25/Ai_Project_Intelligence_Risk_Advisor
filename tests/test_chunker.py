import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.chunker import chunk_text


text = """
Project: Smart Campus Assistant.

The frontend development is almost complete.
The backend authentication system is incomplete.
Database migration is currently delayed.
The deployment deadline is September 20, 2026.
"""


chunks = chunk_text(text, chunk_size=100, overlap=20)

print("----- CHUNKS -----")

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)
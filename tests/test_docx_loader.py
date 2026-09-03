from app.ingestion.docx_loader import load_docx


docx_path = "data/raw/meeting_notes.docx"

text = load_docx(docx_path)

print("----- EXTRACTED DOCX TEXT -----")
print(text)
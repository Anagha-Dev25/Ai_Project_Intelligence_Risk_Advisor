from app.ingestion.pdf_loader import load_pdf


pdf_path = "data/raw/test_project.pdf"

text = load_pdf(pdf_path)

print("----- EXTRACTED TEXT -----")
print(text)
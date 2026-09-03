from app.ingestion.txt_loader import load_txt


txt_path = "data/raw/progress_update.txt"

text = load_txt(txt_path)

print("----- EXTRACTED TXT TEXT -----")
print(text)
def load_markdown(file_path: str) -> str:
    """Extract text from a Markdown (.md) document."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as file:
                return file.read()
        except (UnicodeDecodeError, Exception):
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        return file.read()

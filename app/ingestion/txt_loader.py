def load_txt(file_path: str) -> str:
    """
    Extract text from a plain text file with multi-encoding fallback.

    Args:
        file_path (str): Path to the TXT file.

    Returns:
        str: Extracted and cleaned text.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except (UnicodeDecodeError, Exception):
            continue
            
    # Final fallback
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        return file.read()
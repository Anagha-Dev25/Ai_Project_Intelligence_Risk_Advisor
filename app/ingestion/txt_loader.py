def load_txt(file_path):
    """
    Extract text from a plain text file.

    Args:
        file_path (str): Path to the TXT file.

    Returns:
        str: Extracted text.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text
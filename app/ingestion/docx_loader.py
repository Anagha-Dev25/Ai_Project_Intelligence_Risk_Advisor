from docx import Document


def load_docx(file_path):
    """
    Extract text from a DOCX file.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text from the DOCX.
    """

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text
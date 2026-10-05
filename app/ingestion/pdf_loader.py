from pypdf import PdfReader
import re


def load_pdf(file_path: str) -> str:
    """
    Extract and clean text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted and normalized text from all readable pages.
    """
    reader = PdfReader(file_path)

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise ValueError(f"PDF is encrypted and cannot be opened: {e}")

    extracted_pages = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        
        # Normalize whitespace while preserving line structure
        page_text = re.sub(r"[ \t]+", " ", page_text)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()

        if page_text:
            if len(reader.pages) > 1:
                extracted_pages.append(f"--- [Page {i + 1}] ---\n{page_text}")
            else:
                extracted_pages.append(page_text)

    return "\n\n".join(extracted_pages)
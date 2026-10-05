import re
from docx import Document


def clean_text(text: str) -> str:
    """Normalize whitespace and replace non-standard Word bullet points."""
    if not text:
        return ""
    # Replace common MS Word bullet characters with standard dash
    text = re.sub(r"[\uf0b7\u2022\u25aa\u25cf\u25e6\u2219]", "- ", text)
    # Normalize multiple spaces while preserving newlines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_docx(file_path: str) -> str:
    """
    Extract text and structured table data from a DOCX file.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: Extracted, normalized text from paragraphs and tables.
    """
    document = Document(file_path)
    content_blocks = []

    # 1. Extract paragraphs
    for paragraph in document.paragraphs:
        cleaned = clean_text(paragraph.text)
        if cleaned:
            content_blocks.append(cleaned)

    # 2. Extract tables (enterprise documents heavily use tables for RACI, timelines, risks)
    for table_idx, table in enumerate(document.tables, start=1):
        table_rows = []
        headers = []
        
        for row_idx, row in enumerate(table.rows):
            cells = [clean_text(cell.text) for cell in row.cells]
            # Avoid duplicate adjacent cells in merged rows
            dedup_cells = []
            for c in cells:
                if not dedup_cells or c != dedup_cells[-1]:
                    dedup_cells.append(c)
            
            if not any(dedup_cells):
                continue

            if row_idx == 0:
                headers = dedup_cells
                table_rows.append(" | ".join(dedup_cells))
            else:
                if headers and len(headers) == len(dedup_cells):
                    formatted_row = ", ".join(f"{h}: {val}" for h, val in zip(headers, dedup_cells) if val)
                    table_rows.append(formatted_row)
                else:
                    table_rows.append(" | ".join(dedup_cells))

        if table_rows:
            table_text = f"[Table {table_idx}]\n" + "\n".join(table_rows)
            content_blocks.append(table_text)

    return "\n\n".join(content_blocks)
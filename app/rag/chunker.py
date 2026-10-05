import re
from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into sentence-aware and structure-aware chunks with true sliding overlap.

    Preserves sentence and paragraph boundaries whenever possible, avoids cutting
    through words, and ensures contextual overlap between consecutive chunks for
    reliable RAG retrieval.

    Args:
        text (str): Input text to chunk.
        chunk_size (int): Maximum character length of each chunk.
        overlap (int): Number of overlapping characters to retain between chunks.

    Returns:
        List[str]: List of cleanly formed text chunks.
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    # Preserve markdown headings, lists, and paragraphs by splitting on double newlines
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Break paragraphs into atomic semantic units (sentences or list lines)
    units = []
    for para in raw_paragraphs:
        # Split on sentence boundaries or list line breaks
        sub_units = re.split(r"(?<=[.!?])\s+|\n+", para)
        for u in sub_units:
            u_clean = re.sub(r"\s+", " ", u).strip()
            if not u_clean:
                continue

            # If a single unit exceeds chunk_size, split by word boundaries
            if len(u_clean) > chunk_size:
                words = u_clean.split()
                temp_word_chunk = []
                temp_len = 0
                for w in words:
                    if temp_len + len(w) + 1 > chunk_size and temp_word_chunk:
                        units.append(" ".join(temp_word_chunk))
                        temp_word_chunk = [w]
                        temp_len = len(w)
                    else:
                        temp_word_chunk.append(w)
                        temp_len += len(w) + 1
                if temp_word_chunk:
                    units.append(" ".join(temp_word_chunk))
            else:
                units.append(u_clean)

    if not units:
        return []

    chunks = []
    current_units = []
    current_length = 0

    for unit in units:
        unit_len = len(unit)
        additional_len = unit_len + (1 if current_units else 0)

        if current_length + additional_len <= chunk_size:
            current_units.append(unit)
            current_length += additional_len
        else:
            if current_units:
                chunk_str = " ".join(current_units)
                chunks.append(chunk_str)

                # Build overlap from the end of current_units
                overlap_units = []
                overlap_len = 0
                for prev_unit in reversed(current_units):
                    if overlap_len + len(prev_unit) + (1 if overlap_units else 0) <= overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_len += len(prev_unit) + (1 if len(overlap_units) > 1 else 0)
                    else:
                        break

                current_units = overlap_units + [unit]
                current_length = sum(len(u) for u in current_units) + (len(current_units) - 1)
            else:
                chunks.append(unit)
                current_units = []
                current_length = 0

    if current_units:
        final_str = " ".join(current_units)
        # Avoid adding exact duplicate of previous chunk
        if not chunks or chunks[-1] != final_str:
            chunks.append(final_str)

    return chunks
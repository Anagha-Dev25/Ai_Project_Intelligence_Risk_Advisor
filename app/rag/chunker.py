import re


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into sentence-aware chunks.

    The chunker tries to keep complete sentences together
    instead of cutting words or sentences in half.
    """

    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        current_text = " ".join(current_chunk)

        # If adding the sentence stays within the target size
        if len(current_text) + len(sentence) + 1 <= chunk_size:
            current_chunk.append(sentence)

        else:
            # Save the current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Start a new chunk
            current_chunk = [sentence]

    # Add the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
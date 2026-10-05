import os
from typing import Optional, Tuple
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.txt_loader import load_txt
from app.ingestion.csv_loader import load_csv
from app.ingestion.markdown_loader import load_markdown
from app.ingestion.json_loader import load_json

from app.models.document import Document
from app.rag.chunker import chunk_text
from app.rag.langchain_rag import LangChainRAG
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


def load_raw_text(file_path: str) -> Tuple[str, str]:
    """
    Extract raw text from a document based on its extension.

    Args:
        file_path: Path to the target file.

    Returns:
        Tuple[str, str]: (Extracted text, normalized file extension)
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        text = load_pdf(file_path)
    elif extension == ".docx":
        text = load_docx(file_path)
    elif extension == ".txt":
        text = load_txt(file_path)
    elif extension == ".csv":
        text = load_csv(file_path)
    elif extension == ".md":
        text = load_markdown(file_path)
    elif extension == ".json":
        text = load_json(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{extension}'. Supported types: {SUPPORTED_EXTENSIONS}"
        )

    return text, extension


def process_document(file_path: str, source_name: Optional[str] = None) -> int:
    """
    Load, chunk, and store a project document in ChromaDB using domain models.

    Args:
        file_path (str): Temporary or persistent path of the uploaded document.
        source_name (str, optional): Original uploaded filename.

    Returns:
        int: Number of chunks stored.
    """
    if source_name is None:
        source_name = os.path.basename(file_path)

    # 1. Load document text and type
    text, extension = load_raw_text(file_path)

    # 2. Split text into chunks
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    if not chunks:
        raise ValueError("No readable text was found in the document.")

    # 3. Encapsulate in Document domain model
    doc_model = Document(
        content=text,
        source=source_name,
        file_type=extension.lstrip("."),
        metadata={
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else len(text),
            "char_count": len(text),
            "word_count": len(text.split()),
        },
        chunk_count=len(chunks)
    )

    # 4. Store chunks in ChromaDB through LangChain
    rag = LangChainRAG()
    rag.add_documents(
        chunks,
        source=source_name,
        metadata={
            "file_type": doc_model.file_type,
            "word_count": doc_model.word_count
        }
    )

    return len(chunks)
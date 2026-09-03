import os

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.txt_loader import load_txt
from app.ingestion.csv_loader import load_csv

from app.rag.chunker import chunk_text
from app.rag.langchain_rag import LangChainRAG


def process_document(file_path, source_name=None):
    """
    Load, chunk, and store a project document in ChromaDB.

    Args:
        file_path (str): Temporary path of the uploaded document.
        source_name (str): Original uploaded filename.

    Returns:
        int: Number of chunks stored.
    """

    extension = os.path.splitext(file_path)[1].lower()

    # 1. Load document
    if extension == ".pdf":
        text = load_pdf(file_path)

    elif extension == ".docx":
        text = load_docx(file_path)

    elif extension == ".txt":
        text = load_txt(file_path)

    elif extension == ".csv":
        text = load_csv(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    # 2. Split text into chunks
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No readable text was found in the document.")

    # 3. Use original filename as stable source
    if source_name is None:
        source_name = os.path.basename(file_path)

    # 4. Store chunks in ChromaDB through LangChain
    rag = LangChainRAG()

    rag.add_documents(
        chunks,
        source=source_name
    )

    return len(chunks)
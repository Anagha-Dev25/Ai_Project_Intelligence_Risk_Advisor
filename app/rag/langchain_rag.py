import os
import hashlib
from typing import List, Union, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.rag.langchain_embeddings import SentenceTransformerEmbeddings
from app.config import VECTOR_STORE_DIR, COLLECTION_NAME


class LangChainRAG:
    """
    Enterprise-grade RAG retrieval system backed by ChromaDB and LangChain.
    Supports multi-source scoping, score-ranked similarity search, and collection management.
    """

    def __init__(
        self,
        persist_directory: str = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = SentenceTransformerEmbeddings()

        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(
        self,
        texts: List[str],
        source: str = "uploaded_document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store document chunks and associated metadata with deterministic IDs.

        Args:
            texts: List of text chunk strings.
            source: Source filename or identifier.
            metadata: Optional additional metadata dictionary.

        Returns:
            int: Number of chunks added.
        """
        if not texts:
            return 0

        base_meta = metadata or {}
        documents = []
        ids = []

        for i, text in enumerate(texts):
            doc_meta = {
                "source": source,
                "chunk_id": i,
                **base_meta
            }
            # Deterministic ID preventing collision
            chunk_hash = hashlib.md5(f"{source}_{i}_{text[:50]}".encode()).hexdigest()[:12]
            chunk_id = f"{source}_{i}_{chunk_hash}"

            documents.append(
                Document(
                    page_content=text,
                    metadata=doc_meta
                )
            )
            ids.append(chunk_id)

        self.vectorstore.add_documents(
            documents=documents,
            ids=ids
        )
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
        source: Optional[Union[str, List[str]]] = None
    ) -> List[Document]:
        """
        Retrieve the most relevant document chunks with deduplication and score-ranking.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            source: Single source name, list of source names, or None for all.

        Returns:
            List[Document]: Ranked, deduplicated documents.
        """
        if not query or not query.strip():
            return []

        # If source is a list of sources, search each and rank by similarity score
        if isinstance(source, list):
            valid_sources = [s for s in source if s]
            if not valid_sources:
                return []

            scored_results = []
            for s_name in valid_sources:
                try:
                    res_with_scores = self.vectorstore.similarity_search_with_score(
                        query,
                        k=top_k,
                        filter={"source": s_name}
                    )
                    scored_results.extend(res_with_scores)
                except Exception:
                    # Fallback if filter fails on empty subset
                    continue

            # Sort by score ascending (lower distance is better in Chroma)
            scored_results.sort(key=lambda item: item[1])

            # Deduplicate by content
            unique_docs = []
            seen_texts = set()
            for doc, score in scored_results:
                content = doc.page_content.strip()
                if content not in seen_texts:
                    seen_texts.add(content)
                    unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break

            return unique_docs

        # Single source filter
        if isinstance(source, str) and source.strip():
            try:
                return self.vectorstore.similarity_search(
                    query,
                    k=top_k,
                    filter={"source": source.strip()}
                )
            except Exception:
                return []

        # Global search across all project documents
        try:
            return self.vectorstore.similarity_search(
                query,
                k=top_k
            )
        except Exception:
            return []

    def clear_all(self) -> None:
        """Reset and wipe the vectorstore collection."""
        try:
            self.vectorstore.reset_collection()
        except Exception:
            # Fallback re-init
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            count = self.vectorstore._collection.count()
            return {"total_chunks": count, "collection": self.collection_name}
        except Exception:
            return {"total_chunks": 0, "collection": self.collection_name}
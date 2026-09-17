from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.rag.langchain_embeddings import SentenceTransformerEmbeddings


class LangChainRAG:

    def __init__(self, persist_directory="vector_store/chroma_db"):
        self.embeddings = SentenceTransformerEmbeddings()

        self.vectorstore = Chroma(
            collection_name="project_documents_langchain",
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, texts, source="uploaded_document"):

        documents = [
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "chunk_id": i
                }
            )
            for i, text in enumerate(texts)
        ]

        ids = [
            f"{source}_chunk_{i}"
            for i in range(len(documents))
        ]

        self.vectorstore.add_documents(
            documents=documents,
            ids=ids
        )

    def search(self, query, top_k=3, source=None):

        # If a specific source is provided, search each source separately.
        # This avoids Chroma errors with list-valued filters.
        if isinstance(source, list):

            all_results = []

            for source_name in source:

                results = self.vectorstore.similarity_search(
                    query,
                    k=top_k,
                    filter={"source": source_name}
                )

                all_results.extend(results)

            return all_results[:top_k]

        # Search a single source
        if isinstance(source, str):

            return self.vectorstore.similarity_search(
                query,
                k=top_k,
                filter={"source": source}
            )

        # Search entire project knowledge base
        return self.vectorstore.similarity_search(
            query,
            k=top_k
        )
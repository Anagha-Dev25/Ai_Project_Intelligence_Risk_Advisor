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

        search_kwargs = {
            "k": top_k
        }

        # Search only within the currently uploaded document
        if source:
            search_kwargs["filter"] = {
                "source": source
            }

        retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )

        return retriever.invoke(query)
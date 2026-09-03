import chromadb


class ChromaVectorStore:
    def __init__(self, persist_directory="vector_store/chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="project_documents"
        )

    def add_documents(self, documents, embeddings):
        """
        Store document chunks and their embeddings in ChromaDB.
        """

        ids = [
            f"chunk_{i}"
            for i in range(len(documents))
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

    def search(self, query_embedding, top_k=3):
        """
        Retrieve the most relevant document chunks.
        """

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_documents = results["documents"][0]
        distances = results["distances"][0]

        return [
            {
                "document": document,
                "distance": distance
            }
            for document, distance
            in zip(retrieved_documents, distances)
        ]
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        embeddings = np.asarray(embeddings).astype("float32")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k=3):
        query_embedding = np.asarray(
            [query_embedding]
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index != -1:
                results.append({
                    "document": self.documents[index],
                    "distance": float(distance)
                })

        return results

    def save(self, index_path, documents_path):
        faiss.write_index(self.index, index_path)

        with open(documents_path, "w", encoding="utf-8") as file:
            for document in self.documents:
                file.write(document + "\n")

    def load(self, index_path, documents_path):
        self.index = faiss.read_index(index_path)

        with open(documents_path, "r", encoding="utf-8") as file:
            self.documents = [
                line.rstrip("\n")
                for line in file
            ]
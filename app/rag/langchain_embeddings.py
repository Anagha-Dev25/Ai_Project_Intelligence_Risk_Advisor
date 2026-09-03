from langchain_core.embeddings import Embeddings

from app.rag.embedding_model import EmbeddingModel


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = EmbeddingModel(model_name)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)
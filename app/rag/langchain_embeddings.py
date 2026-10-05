from langchain_core.embeddings import Embeddings
from app.rag.embedding_model import EmbeddingModel
from app.config import EMBEDDING_MODEL_NAME


class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain compatible Embeddings adapter using local sentence-transformers models.
    """

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = EmbeddingModel(model_name)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)
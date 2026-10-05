from sentence_transformers import SentenceTransformer
import threading
from app.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Thread-safe, singleton-cached wrapper for sentence-transformers embedding models.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        with cls._lock:
            if cls._instance is None or cls._instance._model_name != model_name:
                instance = super(EmbeddingModel, cls).__new__(cls)
                instance._model_name = model_name
                instance.model = SentenceTransformer(model_name)
                cls._instance = instance
            return cls._instance

    def embed_text(self, text: str):
        return self.model.encode(text)

    def embed_documents(self, texts: list):
        if not texts:
            return []
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str):
        return self.model.encode(text, show_progress_bar=False).tolist()
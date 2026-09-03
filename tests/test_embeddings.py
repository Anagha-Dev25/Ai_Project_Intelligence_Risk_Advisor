from app.rag.embedding_model import EmbeddingModel


embedding_model = EmbeddingModel()

text = "Authentication is incomplete and delaying integration testing."

embedding = embedding_model.embed_text(text)

print("----- EMBEDDING TEST -----")
print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
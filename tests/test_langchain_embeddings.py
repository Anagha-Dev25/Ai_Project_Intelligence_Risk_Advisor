from app.rag.langchain_embeddings import SentenceTransformerEmbeddings


embedding_model = SentenceTransformerEmbeddings()

text = "Authentication is incomplete."

embedding = embedding_model.embed_query(text)

print("----- LANGCHAIN EMBEDDING TEST -----")
print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
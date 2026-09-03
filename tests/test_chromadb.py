from app.rag.embedding_model import EmbeddingModel
from app.rag.chromadb_store import ChromaVectorStore


embedding_model = EmbeddingModel()

documents = [
    "Authentication is incomplete and delaying integration testing.",
    "The frontend development is almost complete.",
    "The team is planning a vacation next month."
]

embeddings = embedding_model.embed_documents(documents)

vector_store = ChromaVectorStore()

vector_store.add_documents(
    documents,
    embeddings
)

query = "What is blocking integration testing?"

query_embedding = embedding_model.embed_text(query)

results = vector_store.search(
    query_embedding,
    top_k=2
)

print("----- CHROMADB SEARCH RESULTS -----")

for result in results:
    print("\nDocument:", result["document"])
    print("Distance:", result["distance"])
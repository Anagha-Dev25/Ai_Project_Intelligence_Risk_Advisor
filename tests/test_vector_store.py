from app.rag.embedding_model import EmbeddingModel
from app.rag.vector_store import VectorStore


embedding_model = EmbeddingModel()

documents = [
    "Authentication is incomplete and delaying integration testing.",
    "The frontend development is almost complete.",
    "The team is planning a vacation next month."
]

embeddings = embedding_model.embed_documents(documents)

vector_store = VectorStore()

vector_store.add(embeddings, documents)

# Save the vector store
vector_store.save(
    "vector_store/index.faiss",
    "vector_store/documents.txt"
)

print("Vector store saved successfully.")

# Create a new vector store
new_vector_store = VectorStore()

# Load the saved vector store
new_vector_store.load(
    "vector_store/index.faiss",
    "vector_store/documents.txt"
)

query = "What is blocking integration testing?"

query_embedding = embedding_model.embed_text(query)

results = new_vector_store.search(
    query_embedding,
    top_k=2
)

print("\n----- SEARCH AFTER LOADING -----")

for result in results:
    print("\nDocument:", result["document"])
    print("Distance:", result["distance"])
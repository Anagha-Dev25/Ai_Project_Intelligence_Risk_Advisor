from app.rag.langchain_rag import LangChainRAG


documents = [
    "Authentication is incomplete and delaying integration testing.",
    "The frontend development is almost complete.",
    "The project budget has been approved by management."
]

rag = LangChainRAG()

rag.add_documents(documents)

results = rag.search(
    "What is blocking integration testing?",
    top_k=2
)

print("----- LANGCHAIN RAG SEARCH -----")

for i, document in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(document.page_content)
from langchain_ollama import ChatOllama
from app.rag.langchain_rag import LangChainRAG


class BlockerAgent:
    def __init__(self):
        self.rag = LangChainRAG()

        self.llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
            num_predict=250,
             keep_alive="10m"
        )

    def identify_blockers(self, source, top_k=3):

        query = (
            "Find information about blockers, pending decisions, "
            "unresolved issues, action items, dependencies, "
            "pending approvals, missing inputs, assigned tasks, "
            "and responsibilities in this project."
        )

        results = self.rag.search(
            query,
            top_k=top_k,
            source=source
        )

        evidence = [
            document.page_content
            for document in results
        ]

        evidence_text = "\n\n".join(
            f"- {item}" for item in evidence
        )

        prompt = f"""
You are an Enterprise Project Blocker and Action Item Agent.

Analyze ONLY the evidence retrieved from the project document.

Extract the following:

1. BLOCKERS
2. PENDING DECISIONS
3. ACTION ITEMS
4. RESPONSIBLE PERSON

IMPORTANT RULES:

- Use ONLY information supported by the evidence.
- Do not invent blockers or unresolved problems.
- A dependency is not automatically a blocker.
- A planned task is not automatically an action item.
- Do not assume that an approval is pending unless the evidence
  explicitly says it is pending.
- Preserve names, tasks, and dates exactly when available.
- Remove duplicate items.
- If information is not present, write:
  "Not specified in the document."

For ACTION ITEMS, include the responsible person only when
the evidence supports the assignment.

Return the result in this format:

### BLOCKERS
- Item

### PENDING DECISIONS
- Decision

### ACTION ITEMS
- Action — Responsible person

### RESPONSIBLE PERSONS
- Person — Responsibility

Retrieved evidence:

{evidence_text}
"""

        response = self.llm.invoke(prompt)

        return {
            "source": source,
            "analysis": response.content,
            "evidence": evidence
        }
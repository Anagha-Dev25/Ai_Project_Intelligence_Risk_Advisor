from typing import Dict, Any, Union, List
from langchain_ollama import ChatOllama
from app.rag.langchain_rag import LangChainRAG
from app.config import DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_KEEP_ALIVE


class BlockerAgent:
    """
    Enterprise AI Agent for identifying blocked tasks, unresolved dependencies,
    pending architectural/management decisions, and action items from project evidence.
    """

    def __init__(self):
        self.rag = LangChainRAG()
        self.llm = ChatOllama(
            model=DEFAULT_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=600,
            keep_alive=LLM_KEEP_ALIVE
        )

    def identify_blockers(
        self,
        source: Union[str, List[str]],
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Dynamically identify blockers and action items from retrieved project evidence.

        Args:
            source: Document source name or list of sources.
            top_k: Number of evidence chunks to retrieve.

        Returns:
            Dict containing 'source', 'analysis', and 'evidence'.
        """
        query = (
            "Find information about currently blocked work, "
            "unresolved issues, missing credentials, missing inputs, "
            "pending approvals, dependencies preventing progress, action items, "
            "and responsible team members."
        )

        results = self.rag.search(
            query,
            top_k=top_k,
            source=source
        )

        evidence = [doc.page_content for doc in results]
        unique_evidence = list(dict.fromkeys(evidence))

        if not unique_evidence:
            return {
                "source": source,
                "analysis": "### BLOCKERS\nNo active blockers identified.\n\n### PENDING DECISIONS\nNo pending decisions identified.\n\n### ACTION ITEMS\nNo evidence-based action items identified.\n\n### RESPONSIBLE PERSONS\nNo assigned responsibilities identified in retrieved evidence.",
                "evidence": []
            }

        evidence_text = "\n\n".join(
            f"[Evidence {i+1}]:\n{item}"
            for i, item in enumerate(unique_evidence)
        )

        prompt = f"""
You are an enterprise agile project management and blocker resolution specialist.
Analyze ONLY the project evidence below to identify active blockers, pending decisions,
action items, and responsible persons.

STRICT FACT-GROUNDING RULES:
1. BLOCKERS: Only identify an item as a blocker if the evidence EXPLICITLY states that
   work is blocked, prevented, waiting for something, or currently unable to proceed.
   Normal ongoing development is NOT a blocker.
2. PENDING DECISIONS: Only include decisions or approvals explicitly described as pending or awaiting resolution.
3. ACTION ITEMS: Include explicit next steps, follow-ups, or corrective tasks mentioned in the evidence, with owners and deadlines when available.
4. RESPONSIBLE PERSONS: List people and their exact documented responsibilities. Never guess or switch roles.
5. If a section has no evidence, write: "None identified in the provided evidence."

REQUIRED OUTPUT FORMAT (preserve these exact headings):
### BLOCKERS
- [List each explicit blocker, or "No active blockers identified."]

### PENDING DECISIONS
- [List each pending decision or approval, or "None identified."]

### ACTION ITEMS
- [Action item — Owner (if known) — Deadline (if known)]

### RESPONSIBLE PERSONS
- [Name / Role — Documented Responsibility]

==================================================
PROJECT EVIDENCE:
==================================================
{evidence_text}
"""

        try:
            response = self.llm.invoke(prompt)
            analysis = response.content.strip()
        except Exception as e:
            analysis = f"""### BLOCKERS
Temporary LLM error during blocker analysis: {e}

### PENDING DECISIONS
Refer to project evidence.

### ACTION ITEMS
Refer to project evidence.

### RESPONSIBLE PERSONS
Refer to project evidence."""

        return {
            "source": source,
            "analysis": analysis,
            "evidence": unique_evidence
        }
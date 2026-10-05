from typing import Dict, Any, Union, List
from langchain_ollama import ChatOllama
from app.rag.langchain_rag import LangChainRAG
from app.config import DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_KEEP_ALIVE


class RiskAgent:
    """
    Enterprise AI Agent for detecting project risks, delivery threats,
    schedule vulnerabilities, and forecasting project delivery health.
    """

    def __init__(self):
        self.rag = LangChainRAG()
        self.llm = ChatOllama(
            model=DEFAULT_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=450,
            keep_alive=LLM_KEEP_ALIVE
        )

    def analyze_risks(
        self,
        source: Union[str, List[str]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze project delivery risks based strictly on retrieved evidence.

        Args:
            source: Document source name or list of sources.
            top_k: Number of evidence chunks to retrieve.

        Returns:
            Dict containing 'source', 'analysis', and 'evidence'.
        """
        query = (
            "Find all information related to project risks, "
            "delays, unresolved issues, dependencies, blockers, "
            "deadlines, missing inputs, schedule constraints, "
            "and delivery challenges."
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
                "analysis": "DELIVERY FORECAST: ON TRACK\n\nFORECAST REASON: No documented project risks or delivery impediments found in retrieved evidence.\n\nRISKS:\nNo evidence-based risk identified.",
                "evidence": []
            }

        evidence_text = "\n\n".join(
            f"[Evidence {i+1}]:\n{item}"
            for i, item in enumerate(unique_evidence)
        )

        prompt = f"""
You are an enterprise project risk analyst.
Analyze ONLY the project evidence below.

STRICT RISK EVALUATION RULES:
1. Identify potential risks ONLY when the evidence explicitly contains a dependency,
   deadline conflict, schedule constraint, missing input, unresolved issue, or explicit threat.
2. A deadline by itself is NOT a risk.
3. A planned milestone is NOT a risk.
4. A responsibility is NOT a risk.
5. A dependency is a POTENTIAL RISK only when failure to satisfy it could affect delivery.
6. For potential risks, describe the impact using cautious language ("could affect", "may delay").
7. Never state that something IS delayed unless the evidence explicitly says it is delayed.
8. If the evidence contains no actual delay or blocker, use ON TRACK.

REQUIRED OUTPUT FORMAT (preserve this exact format):
DELIVERY FORECAST: ON TRACK / AT RISK / DELAYED

FORECAST REASON: [one short sentence based directly on the evidence]

RISKS:
1. Risk: [Identified risk title/description]
   Reason: [Why this is a risk based on the evidence]
   Impact: [Potential consequence on delivery or quality]
   Mitigation: [Direct, realistic mitigation step]

2. Risk: ...
   Reason: ...
   Impact: ...
   Mitigation: ...

If no risk is supported by the evidence, output:
RISKS:
No evidence-based risk identified.

==================================================
PROJECT EVIDENCE:
==================================================
{evidence_text}
"""

        try:
            response = self.llm.invoke(prompt)
            analysis = response.content.strip()
        except Exception as e:
            analysis = f"DELIVERY FORECAST: AT RISK\n\nFORECAST REASON: Temporary error during risk analysis ({e}).\n\nRISKS:\nRefer to project evidence."

        return {
            "source": source,
            "analysis": analysis,
            "evidence": unique_evidence
        }
from langchain_ollama import ChatOllama
from app.rag.langchain_rag import LangChainRAG
import re

class RiskAgent:
    def __init__(self):
        self.rag = LangChainRAG()

        self.llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
            num_predict=250,
             keep_alive="10m"

        )

    def analyze_risks(self, source, top_k=3):

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

        evidence = [
            document.page_content
            for document in results
        ]

        # Remove duplicate evidence
        risk_evidence = list(dict.fromkeys(evidence))

        evidence_text = "\n\n".join(
            f"- {item}" for item in risk_evidence
        )

        prompt = f"""
You are an enterprise project risk analyst.

Analyze ONLY the evidence below.

Identify at most 3 potential risks ONLY when the evidence
contains a dependency, deadline, constraint, missing input,
unresolved issue, or explicit threat that could affect delivery.

A deadline by itself is NOT a risk.
A planned milestone is NOT a risk.
A responsibility is NOT a risk.
A dependency is not automatically a risk.
Treat a dependency as a POTENTIAL RISK only when the evidence
shows that failure to satisfy it could affect a deadline,
milestone, or delivery activity.
For a POTENTIAL RISK, describe the impact as a possible
future consequence using words such as "could affect",
"may delay", or "could impact".
Never state that something IS delayed unless the evidence
explicitly says it is delayed.
Never describe the dependency as missing, late, or unresolved
unless the evidence explicitly says so.
Do not claim something is delayed unless the evidence says it is delayed.
Do not invent missing credentials, approvals, problems, or delays.

Return ONLY this format:

DELIVERY FORECAST: ON TRACK / AT RISK / DELAYED

FORECAST REASON: one short sentence based directly on the evidence.

If the evidence contains no actual delay, use ON TRACK unless
a documented dependency or constraint creates a clear potential
delivery concern.

Mitigation must be a simple action directly related to the
documented dependency or constraint. Do not invent resources,
budgets, escalations, or approvals.Do not imply that the responsible person is currently behind.

RISKS:
1. Risk: ...
   Reason: ...
   Impact: ...
   Mitigation: ...

2. Risk: ...
   Reason: ...
   Impact: ...
   Mitigation: ...

If no risk is supported, write:

RISKS:
No evidence-based risk identified.

Do not repeat sections.
Do not add extra sections.
Keep the response under 250 words.

EVIDENCE:
{evidence_text}
"""

        response = self.llm.invoke(prompt)

        return {
            "source": source,
            "analysis": response.content,
            "evidence": evidence
        }
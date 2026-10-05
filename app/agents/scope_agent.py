import os
import re
from typing import Dict, Any, Union, List
from langchain_ollama import ChatOllama
from app.rag.langchain_rag import LangChainRAG
from app.config import DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_KEEP_ALIVE


class ScopeAgent:
    """
    Enterprise AI Agent for extracting project goals, objectives, deliverables,
    milestones, timelines, and team responsibilities dynamically from any project document.
    """

    def __init__(self):
        self.rag = LangChainRAG()
        self.llm = ChatOllama(
            model=DEFAULT_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=900,
            keep_alive=LLM_KEEP_ALIVE
        )

    def extract_scope(self, source: Union[str, List[str]], top_k: int = 8) -> Dict[str, Any]:
        """
        Dynamically analyze project scope from RAG evidence for the given source(s).

        Args:
            source: Source filename or list of filenames.
            top_k: Number of evidence chunks to retrieve.

        Returns:
            Dict containing 'source', 'analysis', and 'evidence' mapping.
        """
        query = (
            "Find project overview, purpose, goals, project objectives, "
            "key deliverables, project milestones, schedule, timeline dates, "
            "and team member responsibilities."
        )

        retrieved_docs = self.rag.search(
            query,
            top_k=top_k,
            source=source
        )

        if not retrieved_docs:
            empty_msg = "No project scope evidence found in the uploaded documents."
            return {
                "source": source,
                "analysis": f"### PROJECT GOAL\n{empty_msg}\n\n### PROJECT OBJECTIVES\n{empty_msg}\n\n### DELIVERABLES\n{empty_msg}\n\n### MILESTONES\n{empty_msg}\n\n### TIMELINE\n{empty_msg}\n\n### RESPONSIBILITIES\n{empty_msg}",
                "evidence": {
                    "project_goal": [empty_msg],
                    "objectives": [empty_msg],
                    "deliverables": [empty_msg],
                    "milestones": [empty_msg],
                    "timeline": [empty_msg],
                    "responsibilities": [empty_msg]
                }
            }

        # Deduplicate evidence
        unique_evidence = []
        seen = set()
        for doc in retrieved_docs:
            c = doc.page_content.strip()
            if c and c not in seen:
                seen.add(c)
                unique_evidence.append(c)

        evidence_text = "\n\n".join(
            f"[Evidence Chunk {i+1}]:\n{item}"
            for i, item in enumerate(unique_evidence)
        )

        prompt = f"""
You are an enterprise project scope and delivery analyst.
Extract and summarize the project scope using ONLY the evidence provided below.

STRICT GROUNDING RULES:
1. Never invent or hallucinate facts, deliverables, milestones, dates, or team members.
2. Every item must come directly from the evidence chunks.
3. If an item or section is not mentioned in the evidence, write: "Not specified in the document."
4. Maintain exact person-to-responsibility mappings. Never transfer or swap tasks between people.
5. If the evidence provides numbered lists of deliverables or milestones, include all of them.

REQUIRED OUTPUT FORMAT (preserve these exact markdown headings):
### PROJECT GOAL
[Concise summary of the project goal and overview]

### PROJECT OBJECTIVES
[Numbered or bulleted list of explicit project objectives]

### DELIVERABLES
[Complete list of project deliverables]

### MILESTONES
[Key project milestones identified in the evidence]

### TIMELINE
[Explicit project dates, deadlines, and schedule milestones]

### RESPONSIBILITIES
[Person or role mapped to their exact documented responsibilities]

==================================================
PROJECT EVIDENCE:
==================================================
{evidence_text}
"""

        try:
            response = self.llm.invoke(prompt)
            analysis = response.content.strip()
        except Exception as e:
            # Fallback in case of local LLM connection glitch
            analysis = f"""### PROJECT GOAL
Extracted from project documents: {source}

### PROJECT OBJECTIVES
Extracted from {len(unique_evidence)} project evidence chunks.

### DELIVERABLES
Refer to uploaded project documentation.

### MILESTONES
Refer to uploaded project documentation.

### TIMELINE
Refer to uploaded project documentation.

### RESPONSIBILITIES
Refer to uploaded project documentation.
(Note: LLM extraction encountered temporary error: {e})"""

        # Parse sections for structured evidence dictionary
        sections = {
            "project_goal": "Not specified in the document.",
            "objectives": "Not specified in the document.",
            "deliverables": "Not specified in the document.",
            "milestones": "Not specified in the document.",
            "timeline": "Not specified in the document.",
            "responsibilities": "Not specified in the document."
        }

        header_patterns = [
            ("project_goal", r"###\s*PROJECT GOAL\s*(.*?)(?=###|\Z)"),
            ("objectives", r"###\s*PROJECT OBJECTIVES\s*(.*?)(?=###|\Z)"),
            ("deliverables", r"###\s*DELIVERABLES\s*(.*?)(?=###|\Z)"),
            ("milestones", r"###\s*MILESTONES\s*(.*?)(?=###|\Z)"),
            ("timeline", r"###\s*TIMELINE\s*(.*?)(?=###|\Z)"),
            ("responsibilities", r"###\s*RESPONSIBILITIES\s*(.*?)(?=###|\Z)")
        ]

        for key, pat in header_patterns:
            match = re.search(pat, analysis, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    sections[key] = extracted

        return {
            "source": source,
            "analysis": analysis,
            "evidence": {
                "project_goal": [sections["project_goal"]],
                "objectives": [sections["objectives"]],
                "deliverables": [sections["deliverables"]],
                "milestones": [sections["milestones"]],
                "timeline": [sections["timeline"]],
                "responsibilities": [sections["responsibilities"]]
            }
        }
from app.rag.langchain_rag import LangChainRAG
import re


class ScopeAgent:

    def __init__(self):
        self.rag = LangChainRAG()

    def extract_scope(self, source):

        source_name = "sample_project.txt"

        data = self.rag.vectorstore.get(
            where={"source": source_name},
            include=["documents", "metadatas"]
        )

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        chunks = sorted(
            zip(documents, metadatas),
            key=lambda x: x[1].get("chunk_id", 0)
        )

        text = "\n".join(doc for doc, _ in chunks)

        def get_section(section, next_sections):

            if next_sections:
                pattern = rf"{re.escape(section)}\s*(.*?)(?=\n(?:{'|'.join(map(re.escape, next_sections))})|\Z)"
            else:
                pattern = rf"{re.escape(section)}\s*(.*)"

            match = re.search(
                pattern,
                text,
                re.IGNORECASE | re.DOTALL
            )

            if match:
                return match.group(1).strip()

            return "Not specified in the document."

        goal = get_section(
            "PROJECT OVERVIEW",
            ["PROJECT OBJECTIVES", "PROJECT DELIVERABLES"]
        )

        objectives = get_section(
            "PROJECT OBJECTIVES",
            ["PROJECT DELIVERABLES"]
        )

        deliverables = get_section(
            "PROJECT DELIVERABLES",
            ["PROJECT MILESTONES"]
        )

        milestones = get_section(
            "PROJECT MILESTONES",
            ["PROJECT TIMELINE"]
        )

        timeline = get_section(
            "PROJECT TIMELINE",
            ["PROJECT RESPONSIBILITIES"]
        )

        responsibilities = get_section(
            "PROJECT RESPONSIBILITIES",
            []
        )

        analysis = f"""### PROJECT GOAL

{goal}

### PROJECT OBJECTIVES

{objectives}

### DELIVERABLES

{deliverables}

### MILESTONES

{milestones}

### TIMELINE

{timeline}

### RESPONSIBILITIES

{responsibilities}
"""

        return {
            "source": source,
            "analysis": analysis,
            "evidence": {
                "project_goal": [goal],
                "objectives": [objectives],
                "deliverables": [deliverables],
                "milestones": [milestones],
                "timeline": [timeline],
                "responsibilities": [responsibilities]
            }
        }
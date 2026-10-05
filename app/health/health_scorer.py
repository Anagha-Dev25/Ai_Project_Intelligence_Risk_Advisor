import re
from typing import Dict, Any, Union, List
from langchain_ollama import ChatOllama
from app.rag.langchain_rag import LangChainRAG
from app.config import DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_KEEP_ALIVE


class ProjectHealthScorer:
    """
    Enterprise health scoring engine calculating multi-dimensional project health
    (Scope Clarity, Timeline Health, Risk Status, Blocker Status) based on evidence.
    """

    def __init__(self):
        self.rag = LangChainRAG()
        self.llm = ChatOllama(
            model=DEFAULT_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=800,
            keep_alive=LLM_KEEP_ALIVE
        )

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def _retrieve(self, query: str, source: Union[str, List[str]], top_k: int = 5) -> List[str]:
        results = self.rag.search(
            query,
            top_k=top_k,
            source=source
        )

        all_evidence = []
        for document in results:
            content = document.page_content.strip()
            if content and content not in all_evidence:
                all_evidence.append(content)

        return all_evidence

    # =========================================================
    # RISK / BLOCKER CLASSIFICATION
    # =========================================================

    def _classify_evidence(self, evidence: List[str]) -> Dict[str, List[str]]:
        if not evidence:
            return {"risks": [], "blockers": []}

        evidence_text = "\n\n".join(
            f"EVIDENCE {i + 1}:\n{item}"
            for i, item in enumerate(evidence)
        )

        prompt = f"""
You are an enterprise project health evidence classifier.
Analyze ONLY the evidence below.

Identify:
1. EXPLICIT RISKS
2. EXPLICIT BLOCKERS

STRICT CLASSIFICATION RULES:
- A risk must be supported by evidence showing:
  unresolved issue, missing input, dependency affecting delivery, delivery threat,
  pending approval affecting progress, or documented delay.
- Do NOT classify normal planned activities, responsibilities, or milestones as risks.
- A blocker must be explicitly described as something preventing work from proceeding,
  blocking an activity, or causing work to wait.

REQUIRED OUTPUT FORMAT:
RISKS:
- Risk: [Description of risk]

BLOCKERS:
- Blocker: [Description of blocker]

If none identified, write:
RISKS:
NONE

BLOCKERS:
NONE

EVIDENCE:
{evidence_text}
"""

        try:
            response = self.llm.invoke(prompt)
            text = response.content.strip()
        except Exception:
            text = "RISKS:\nNONE\n\nBLOCKERS:\nNONE"

        risks = []
        blockers = []
        current_section = None

        for line in text.splitlines():
            line = line.strip()
            if line.upper() == "RISKS:":
                current_section = "risks"
                continue
            if line.upper() == "BLOCKERS:":
                current_section = "blockers"
                continue
            if line.upper() == "NONE":
                continue

            if current_section == "risks":
                if line.startswith("- Risk:"):
                    r = line.replace("- Risk:", "", 1).strip()
                    if r and r not in risks:
                        risks.append(r)
                elif line.startswith("-") or line.startswith("*"):
                    r = line.lstrip("-* ").strip()
                    if r and r.upper() != "NONE" and r not in risks:
                        risks.append(r)

            elif current_section == "blockers":
                if line.startswith("- Blocker:"):
                    b = line.replace("- Blocker:", "", 1).strip()
                    if b and b not in blockers:
                        blockers.append(b)
                elif line.startswith("-") or line.startswith("*"):
                    b = line.lstrip("-* ").strip()
                    if b and b.upper() != "NONE" and b not in blockers:
                        blockers.append(b)

        return {
            "risks": risks,
            "blockers": blockers
        }

    # =========================================================
    # MAIN HEALTH CALCULATION
    # =========================================================

    def calculate_health(self, source: Union[str, List[str]]) -> Dict[str, Any]:
        # 1. Scope evidence
        scope_evidence = self._retrieve(
            "Find explicit project scope information including overview, goals, "
            "objectives, requirements, deliverables, features, and milestones.",
            source,
            top_k=5
        )

        # 2. Timeline evidence
        timeline_evidence = self._retrieve(
            "Find explicit project timeline, milestones, deadlines, start dates, "
            "end dates, schedule, progress, delivery dates, and delays.",
            source,
            top_k=5
        )

        # 3. Risk and blocker evidence
        risk_blocker_evidence = self._retrieve(
            "Find explicit evidence about project risks, delays, unresolved issues, "
            "blocked work, missing inputs, pending approvals, and schedule constraints.",
            source,
            top_k=6
        )

        # 4. Classify risks & blockers
        classification = self._classify_evidence(risk_blocker_evidence)
        risks = classification["risks"]
        blockers = classification["blockers"]

        # 5. Score dimensions
        scope_score = self._calculate_scope_score(scope_evidence)
        timeline_score = self._calculate_timeline_score(timeline_evidence)
        risk_score = self._calculate_risk_score(risks)
        blocker_score = self._calculate_blocker_score(blockers)

        overall_score = round(
            (
                scope_score * 0.25
                + timeline_score * 0.25
                + risk_score * 0.25
                + blocker_score * 0.25
            ),
            2
        )

        status = self._get_status(overall_score)

        # 6. Health factors
        health_factors = []
        if scope_score >= 85:
            health_factors.append("Project scope is comprehensively documented with clear deliverables and objectives.")
        elif scope_score >= 65:
            health_factors.append("Project scope is defined but could benefit from more detailed specifications.")
        else:
            health_factors.append("Project scope evidence is minimal or incomplete.")

        if timeline_score >= 85:
            health_factors.append("Project timeline contains defined milestones and explicit completion dates.")
        elif timeline_score >= 70:
            health_factors.append("A project timeline is documented, with moderate milestone tracking.")
        else:
            health_factors.append("Timeline evidence lacks explicit dates, milestones, or contains schedule threats.")

        if risks:
            health_factors.append(f"{len(risks)} active project risk(s) identified requiring mitigation.")
        else:
            health_factors.append("No critical delivery risks detected.")

        if blockers:
            health_factors.append(f"{len(blockers)} active blocker(s) impeding immediate progress.")
        else:
            health_factors.append("No active blockers impeding project progress.")

        return {
            "source": source,
            "overall_score": overall_score,
            "status": status,
            "dimensions": {
                "scope_clarity": scope_score,
                "timeline_health": timeline_score,
                "risk_status": risk_score,
                "blocker_status": blocker_score
            },
            "identified_risks": risks,
            "identified_blockers": blockers,
            "health_factors": health_factors,
            "evidence": {
                "scope": scope_evidence,
                "timeline": timeline_evidence,
                "risk_blocker_retrieval": risk_blocker_evidence
            }
        }

    # =========================================================
    # SCORING DIMENSIONS
    # =========================================================

    def _calculate_scope_score(self, evidence: List[str]) -> int:
        if not evidence:
            return 40

        text = " ".join(evidence).lower()
        score = 40

        indicators = [
            "project overview", "project goal", "project objectives",
            "objectives", "requirements", "project deliverables",
            "deliverables", "project milestones", "milestones",
            "features", "scope", "architecture"
        ]

        found = sum(1 for ind in indicators if ind in text)
        score += min(found * 7, 60)
        return min(score, 100)

    def _calculate_timeline_score(self, evidence: List[str]) -> int:
        if not evidence:
            return 40

        text = " ".join(evidence).lower()
        score = 40

        # General schedule indicators
        if any(term in text for term in ["timeline", "schedule", "roadmap"]):
            score += 12

        if any(term in text for term in ["milestone", "milestones", "phase", "sprint"]):
            score += 10

        # Generalized date and temporal recognition
        date_pattern = re.compile(
            r"\b(19|20)\d{2}\b|"  # 4-digit years
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|"
            r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b|"
            r"\b(q[1-4]|quarter\s*[1-4]|sprint\s*\d+|week\s*\d+)\b|"
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
            r"\b(deadline|completion|scheduled|planned|target date|launch)\b",
            re.IGNORECASE
        )

        matches = len(set(match.group(0).lower() for match in date_pattern.finditer(text)))
        score += min(matches * 6, 38)

        return min(score, 100)

    def _calculate_risk_score(self, risks: List[str]) -> int:
        count = len(risks)
        if count == 0:
            return 100
        if count == 1:
            return 80
        if count == 2:
            return 65
        if count == 3:
            return 50
        return 35

    def _calculate_blocker_score(self, blockers: List[str]) -> int:
        count = len(blockers)
        if count == 0:
            return 100
        if count == 1:
            return 60
        if count == 2:
            return 35
        if count == 3:
            return 20
        return 10

    def _get_status(self, score: float) -> str:
        if score >= 85:
            return "HEALTHY"
        if score >= 70:
            return "STABLE"
        if score >= 50:
            return "AT RISK"
        return "CRITICAL"
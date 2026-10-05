from langchain_ollama import ChatOllama

from app.rag.langchain_rag import LangChainRAG


class DocumentationAgent:

    def __init__(self):

        self.rag = LangChainRAG()

        self.llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
            num_predict=600,
            keep_alive="10m"
        )
    # ============================================================
    # RETRIEVE PROJECT EVIDENCE
    # ============================================================

    def _retrieve_evidence(self, sources):

            queries = [
                "project objectives requirements deliverables capabilities features",
                "current risks blockers dependencies pending approvals delays",
                "action items owners deadlines tasks follow up next steps"
            ]
        
            evidence = []
            seen = set()
        
            if isinstance(sources, str):
                sources = [sources]
        
            for query in queries:
        
                try:
                    # Search ALL active documents in one RAG call
                    results = self.rag.search(
                        query,
                        top_k=3,
                        source=sources
                    )
        
                    for doc in results:
        
                        content = doc.page_content.strip()
        
                        if not content:
                            continue
        
                        key = content.lower()
        
                        if key not in seen:
        
                            seen.add(key)
        
                            evidence.append({
                                "source": doc.metadata.get(
                                    "source",
                                    "Unknown"
                                ),
                                "content": content
                            })
        
                except Exception as e:
        
                    print(
                        f"Retrieval error: {e}"
                    )

            return evidence

    # ============================================================
    # FORMAT EVIDENCE
    # ============================================================

    def _format_evidence(self, evidence):

         if not evidence:
             return "NO EVIDENCE FOUND."
     
         formatted = []
         total_chars = 0
     
         # Keep the prompt reasonably small for local Qwen
         MAX_TOTAL_CHARS = 14000
         MAX_CHARS_PER_CHUNK = 2500
     
         for i, item in enumerate(evidence, 1):
     
             content = item["content"]
     
             # Prevent one huge document chunk
             if len(content) > MAX_CHARS_PER_CHUNK:
                 content = content[:MAX_CHARS_PER_CHUNK] + "\n[Evidence truncated]"
     
             block = f"""
EVIDENCE {i}
SOURCE: {item["source"]}
     
{content}
"""     
     
             if total_chars + len(block) > MAX_TOTAL_CHARS:
                 break
     
             formatted.append(block)
             total_chars += len(block)
     
         return "\n-----------------------------\n".join(
             formatted
         )

    # ============================================================
    # CLEAN OUTPUT
    # ============================================================

    def _clean_output(self, text):

        if not text:
            return ""

        text = text.strip()

        text = text.replace(
            "```markdown",
            ""
        )

        text = text.replace(
            "```text",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        return text.strip()

    # ============================================================
    # GENERATE DOCUMENTATION
    # ============================================================

    def generate_documentation(self, sources):

        if isinstance(sources, str):
            sources = [sources]

        if not sources:

            return {
                "user_stories":
                    "No project documents available.",

                "risk_register":
                    "No project documents available.",

                "action_items":
                    "No project documents available."
            }

        print("\n======================================")
        print("DOCUMENTATION INTELLIGENCE")
        print("======================================")

        # --------------------------------------------------------
        # ONE RETRIEVAL PASS
        # --------------------------------------------------------

        print("Collecting project evidence...")

        evidence = self._retrieve_evidence(
            sources
        )

        evidence_text = self._format_evidence(
            evidence
        )

        # --------------------------------------------------------
        # ONE LLM CALL
        # --------------------------------------------------------

        print("Generating documentation...")

        prompt = f"""
You are an enterprise project intelligence analyst.

Analyze ONLY the project evidence provided below.

PROJECT EVIDENCE
================

{evidence_text}

Your task is to generate THREE independent sections:

1. USER STORIES
2. RISK REGISTER
3. ACTION ITEMS

============================================================
SECTION 1 — USER STORIES
============================================================

Generate 3 to 5 useful user stories.

Use ONLY explicit project objectives, requirements,
deliverables, capabilities or features.

Do NOT convert ordinary tasks, deadlines or meetings
into user stories.

Format:

USER STORY 1
Role: <real role>
Goal: <specific goal>
Benefit: <specific benefit>
Evidence: <source>

USER STORY 2
Role: <real role>
Goal: <specific goal>
Benefit: <specific benefit>
Evidence: <source>

============================================================
SECTION 2 — RISK REGISTER
============================================================

Identify CURRENT risks supported by the evidence.

A valid risk must represent an unresolved issue,
missing dependency, pending approval, blocker,
delay or threat to project delivery.

Do NOT invent risks.

Format:

RISK 1
Risk: <specific risk>
Impact: <specific impact>
Owner: <owner or Not specified>
Mitigation: <documented mitigation or Not specified>
Evidence: <source>

RISK 2
Risk: <specific risk>
Impact: <specific impact>
Owner: <owner or Not specified>
Mitigation: <documented mitigation or Not specified>
Evidence: <source>

============================================================
SECTION 3 — ACTION ITEMS
============================================================

Extract explicit current actions and follow-ups.

Preserve the actual owner, deadline and status whenever
they are available in the evidence.

Do NOT invent actions.

Format:

ACTION ITEM 1
Action: <specific action>
Owner: <owner or Not specified>
Deadline: <deadline or Not specified>
Status: <status or Not specified>
Evidence: <source>

ACTION ITEM 2
Action: <specific action>
Owner: <owner or Not specified>
Deadline: <deadline or Not specified>
Status: <status or Not specified>
Evidence: <source>

============================================================
IMPORTANT RULES
============================================================

- Use ONLY the supplied evidence.
- Never invent information.
- Never use square-bracket placeholders.
- Keep each item concise.
- Keep the three sections completely separate.
- Do not merge Risk Register with Action Items.
- Every risk must have evidence.
- Every action item must have evidence.
- If no evidence exists for a section, explicitly say:

NO EVIDENCE-BASED ITEMS IDENTIFIED.

Return all THREE sections.
"""

        try:

            response = self.llm.invoke(
                prompt
            )

            output = self._clean_output(
                response.content
            )

        except Exception as e:

            print(
                f"Documentation generation error: {e}"
            )

            return {
                "user_stories":
                    "Unable to generate user stories.",

                "risk_register":
                    "Unable to generate risk register.",

                "action_items":
                    "Unable to generate action items."
            }

        # ========================================================
        # SPLIT THE SINGLE LLM RESPONSE
        # ========================================================

        user_stories = ""
        risk_register = ""
        action_items = ""

        upper_output = output.upper()

        user_marker = "SECTION 1"
        risk_marker = "SECTION 2"
        action_marker = "SECTION 3"

        # --------------------------------------------------------
        # Find section positions
        # --------------------------------------------------------

        user_pos = upper_output.find(
            user_marker
        )

        risk_pos = upper_output.find(
            risk_marker
        )

        action_pos = upper_output.find(
            action_marker
        )

        # --------------------------------------------------------
        # Extract sections
        # --------------------------------------------------------

        if user_pos != -1:

            start = user_pos

            end = (
                risk_pos
                if risk_pos != -1
                else len(output)
            )

            user_stories = output[
                start:end
            ].strip()

        if risk_pos != -1:

            start = risk_pos

            end = (
                action_pos
                if action_pos != -1
                else len(output)
            )

            risk_register = output[
                start:end
            ].strip()

        if action_pos != -1:

            action_items = output[
                action_pos:
            ].strip()

        # --------------------------------------------------------
        # Fallback if model omitted SECTION labels
        # --------------------------------------------------------

        if not user_stories:

            user_stories = (
                "No user stories generated."
            )

        if not risk_register:

            risk_register = (
                "No evidence-based risks identified."
            )

        if not action_items:

            action_items = (
                "No evidence-based action items identified."
            )

        print("Documentation generation completed.")

        return {
            "user_stories": user_stories,
            "risk_register": risk_register,
            "action_items": action_items
        }
import unittest
import os
import sys
import tempfile

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.document import Document, RiskItem, BlockerItem, ActionItem, UserStory
from app.rag.chunker import chunk_text
from app.ingestion.txt_loader import load_txt
from app.ingestion.csv_loader import load_csv
from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.health.health_scorer import ProjectHealthScorer
from app.export import (
    export_risk_register_csv,
    export_action_items_csv,
    export_user_stories_csv,
    generate_executive_report_markdown
)
from app.config import check_ollama_status, DEFAULT_LLM_MODEL


class TestDocumentModel(unittest.TestCase):
    """Test domain model functionality and properties."""

    def test_document_creation_and_properties(self):
        doc = Document(
            content="This is a test document with eight words.",
            source="test.txt",
            file_type="txt",
            metadata={"priority": "high"}
        )
        self.assertEqual(doc.source, "test.txt")
        self.assertEqual(doc.file_type, "txt")
        self.assertEqual(doc.word_count, 8)
        self.assertEqual(doc.char_count, len(doc.content))

        # Dict conversion
        d = doc.to_dict()
        self.assertEqual(d["source"], "test.txt")
        restored = Document.from_dict(d)
        self.assertEqual(restored.content, doc.content)

    def test_domain_dataclasses(self):
        risk = RiskItem(risk="Credential delay", severity="High", source="plan.pdf")
        self.assertEqual(risk.risk, "Credential delay")
        self.assertEqual(risk.severity, "High")

        blocker = BlockerItem(blocker="Waiting on API keys", owner="Arjun")
        self.assertEqual(blocker.owner, "Arjun")

        action = ActionItem(action="Review architecture", deadline="Oct 20")
        self.assertEqual(action.deadline, "Oct 20")

        story = UserStory(role="Customer", goal="checkout seamlessly", benefit="save time")
        self.assertIn("As a Customer", story.text)


class TestChunker(unittest.TestCase):
    """Test text chunking with semantic boundaries and sliding overlap."""

    def test_empty_and_whitespace(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("Sample text", chunk_size=100, overlap=100)

    def test_sentence_boundary_preservation(self):
        text = "First sentence here. Second sentence is here. Third sentence follows."
        chunks = chunk_text(text, chunk_size=60, overlap=10)
        self.assertTrue(len(chunks) >= 2)
        # Ensure chunks don't cut words awkwardly
        for c in chunks:
            self.assertFalse(c.endswith(" sentenc"))

    def test_sliding_overlap_applied(self):
        text = "Alpha sentence is first. Beta sentence is second. Gamma sentence is third. Delta sentence is fourth."
        chunks = chunk_text(text, chunk_size=55, overlap=30)
        self.assertTrue(len(chunks) >= 2)
        # Check that overlap carries over content
        combined_overlap = any(
            "Beta" in chunks[i] and "Beta" in chunks[i+1]
            for i in range(len(chunks) - 1)
        ) or any(
            "Gamma" in chunks[i] and "Gamma" in chunks[i+1]
            for i in range(len(chunks) - 1)
        )
        self.assertTrue(combined_overlap)


class TestIngestionLoaders(unittest.TestCase):
    """Test loaders against repository sample files."""

    def test_load_txt(self):
        path = os.path.join("data", "raw", "sample_project.txt")
        if os.path.exists(path):
            text = load_txt(path)
            self.assertIn("PROJECT: E-Commerce Platform Modernization", text)
            self.assertIn("PROJECT DELIVERABLES", text)

    def test_load_csv(self):
        path = os.path.join("data", "raw", "project_tasks.csv")
        if os.path.exists(path):
            text = load_csv(path)
            self.assertIn("Cloud architecture design", text)
            self.assertIn("Ananya", text)
            self.assertIn("SUMMARY TABLE", text)
            self.assertIn("STRUCTURED RECORDS", text)

    def test_load_docx(self):
        path = os.path.join("data", "raw", "meeting_notes.docx")
        if os.path.exists(path):
            text = load_docx(path)
            self.assertIn("PROJECT MEETING NOTES", text)
            self.assertIn("Ananya", text)
            self.assertNotIn("\uf0b7", text)  # Bullet characters cleaned

    def test_load_pdf(self):
        path = os.path.join("data", "raw", "test_project.pdf")
        if os.path.exists(path):
            text = load_pdf(path)
            self.assertIn("E-COMMERCE PLATFORM MODERNIZATION", text)
            self.assertIn("Payment Provider Credentials", text)


class TestHealthScorer(unittest.TestCase):
    """Test health scoring calculations and date pattern matching."""

    def setUp(self):
        # We can test internal scoring formulas directly without calling the LLM
        self.scorer = ProjectHealthScorer()

    def test_status_thresholds(self):
        self.assertEqual(self.scorer._get_status(90), "HEALTHY")
        self.assertEqual(self.scorer._get_status(75), "STABLE")
        self.assertEqual(self.scorer._get_status(60), "AT RISK")
        self.assertEqual(self.scorer._get_status(40), "CRITICAL")

    def test_risk_scoring(self):
        self.assertEqual(self.scorer._calculate_risk_score([]), 100)
        self.assertEqual(self.scorer._calculate_risk_score(["Risk 1"]), 80)
        self.assertEqual(self.scorer._calculate_risk_score(["Risk 1", "Risk 2"]), 65)
        self.assertEqual(self.scorer._calculate_risk_score(["R1", "R2", "R3"]), 50)
        self.assertEqual(self.scorer._calculate_risk_score(["R1", "R2", "R3", "R4"]), 35)

    def test_blocker_scoring(self):
        self.assertEqual(self.scorer._calculate_blocker_score([]), 100)
        self.assertEqual(self.scorer._calculate_blocker_score(["Blocker 1"]), 60)
        self.assertEqual(self.scorer._calculate_blocker_score(["B1", "B2"]), 35)

    def test_timeline_dynamic_date_recognition(self):
        # Verify recognition of various dates (2025, 2027, Q3, Sprint 4, March)
        evidence_with_dates = [
            "Project kickoff in March 2027 with Sprint 3 delivery scheduled for Q4.",
            "Deadline is planned for 15/11/2027."
        ]
        score = self.scorer._calculate_timeline_score(evidence_with_dates)
        self.assertGreater(score, 60)


class TestExportUtilities(unittest.TestCase):
    """Test CSV and Markdown report generators."""

    def test_export_risk_register_csv(self):
        sample_doc = {
            "risk_register": """
1. Risk: Payment gateway credential delay
   Reason: External vendor dependency
   Impact: May delay integration testing
   Probability: High
   Severity: High
   Mitigation: Expedite vendor contract
   Evidence: Vendor meeting notes
"""
        }
        csv_str = export_risk_register_csv(sample_doc)
        self.assertIn("Risk ID,Risk Title,Reason,Impact,Probability,Severity,Mitigation,Evidence", csv_str)
        self.assertIn("RSK-001", csv_str)
        self.assertIn("Payment gateway credential delay", csv_str)

    def test_export_action_items_csv(self):
        sample_doc = {
            "action_items": """
1. Action: Finish auth module
   Owner: Meera
   Deadline: October 15, 2026
   Priority: High
   Evidence: Section 4.1
"""
        }
        csv_str = export_action_items_csv(sample_doc)
        self.assertIn("Action ID,Action Item,Owner,Deadline,Priority,Evidence", csv_str)
        self.assertIn("ACT-001", csv_str)
        self.assertIn("Meera", csv_str)

    def test_generate_executive_report_markdown(self):
        report = generate_executive_report_markdown(
            ["sample_project.txt"],
            health_results={"overall_score": 82.5, "status": "STABLE", "dimensions": {"scope_clarity": 90, "timeline_health": 85, "risk_status": 80, "blocker_status": 75}, "health_factors": ["Scope is well defined."]},
            analysis_results=None,
            documentation_results=None
        )
        self.assertIn("# 📊 ProjectIQ Executive Project Intelligence Report", report)
        self.assertIn("82.5/100", report)
        self.assertIn("STABLE", report)


class TestConfiguration(unittest.TestCase):
    """Test configuration module and diagnostics."""

    def test_ollama_status_function(self):
        is_ok, msg, details = check_ollama_status()
        self.assertIsInstance(is_ok, bool)
        self.assertIsInstance(msg, str)


if __name__ == "__main__":
    unittest.main()

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import datetime


@dataclass
class Document:
    """
    Core project document domain model representing ingested files and their metadata.
    """
    content: str
    source: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            content=data.get("content", ""),
            source=data.get("source", "unknown"),
            file_type=data.get("file_type", "unknown"),
            metadata=data.get("metadata", {}),
            chunk_count=data.get("chunk_count", 0),
            created_at=data.get("created_at", "")
        )


@dataclass
class RiskItem:
    """Structured representation of an identified project risk."""
    risk: str
    reason: str = ""
    impact: str = ""
    mitigation: str = ""
    severity: str = "Medium"
    probability: str = "Medium"
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlockerItem:
    """Structured representation of an identified project blocker."""
    blocker: str
    impact: str = ""
    owner: str = "Unassigned"
    status: str = "Active"
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionItem:
    """Structured action item tracked for resolution."""
    action: str
    owner: str = "Unassigned"
    deadline: str = "Not specified"
    priority: str = "Medium"
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserStory:
    """Agile user story generated from project requirements."""
    role: str
    goal: str
    benefit: str
    priority: str = "Medium"
    evidence: str = ""

    @property
    def text(self) -> str:
        return f"As a {self.role}, I want {self.goal}, so that {self.benefit}."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
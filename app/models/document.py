from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    content: str
    source: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
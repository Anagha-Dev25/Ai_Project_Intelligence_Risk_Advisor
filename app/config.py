import os
import urllib.request
import json
from typing import Tuple, Dict, Any

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# Ollama / LLM Settings
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "10m")

# Embedding Settings
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384

# Vector Store Settings
VECTOR_STORE_DIR = os.environ.get(
    "VECTOR_STORE_DIR",
    os.path.join("vector_store", "chroma_db")
)
COLLECTION_NAME = os.environ.get(
    "CHROMA_COLLECTION_NAME",
    "project_documents_langchain"
)

# Text Chunking Settings
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# Supported Ingestion Extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".md", ".json"]


def check_ollama_status() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if the Ollama service is reachable and if the required model is loaded.
    
    Returns:
        (is_connected, message, details)
    """
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "ProjectIQ/1.0"})
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                full_models = [m.get("name", "") for m in data.get("models", [])]
                
                target_base = DEFAULT_LLM_MODEL.split(":")[0]
                has_model = (
                    DEFAULT_LLM_MODEL in full_models or
                    target_base in models or
                    any(target_base in name for name in full_models)
                )
                
                if has_model:
                    return True, f"Ollama connected ({DEFAULT_LLM_MODEL})", data
                else:
                    return (
                        False,
                        f"Ollama connected, but model '{DEFAULT_LLM_MODEL}' not found. Available: {full_models}",
                        data
                    )
            return False, f"Ollama returned HTTP status {response.status}", {}
    except Exception as e:
        return False, f"Could not connect to Ollama at {OLLAMA_BASE_URL}: {e}", {}

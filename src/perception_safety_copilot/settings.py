"""Central configuration for retrieval and local evaluation embeddings."""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _collection_suffix(model_name: str) -> str:
    """Keep vectors from different embedding models in separate collections."""
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")[-24:]
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


@dataclass(frozen=True)
class Settings:
    project1_mcp_enabled: bool = _env_bool("PROJECT1_MCP_ENABLED", True)
    project1_mcp_project_dir: Path = _project_path(
        os.getenv("PROJECT1_MCP_PROJECT_DIR", "../Autonomous_Driving_Safety_Analyst")
    )
    project1_mcp_server: str = os.getenv("PROJECT1_MCP_SERVER", "mcp_server.py")
    project1_mcp_timeout: int = int(os.getenv("PROJECT1_MCP_TIMEOUT", "90"))
    project1_mcp_embedding_backend: str = os.getenv(
        "PROJECT1_MCP_EMBEDDING_BACKEND", "local"
    ).strip().lower()
    project1_results_per_source: int = int(os.getenv("PROJECT1_RESULTS_PER_SOURCE", "3"))
    project1_mcp_video_enabled: bool = _env_bool("PROJECT1_MCP_VIDEO_ENABLED", False)
    lexical_fallback_enabled: bool = _env_bool("LEXICAL_FALLBACK_ENABLED", True)

    local_embeddings_enabled: bool = _env_bool("LOCAL_EMBEDDINGS_ENABLED", True)
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    local_embedding_device: str = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")
    evaluation_chroma_path: Path = _project_path(
        os.getenv("EVALUATION_CHROMA_PATH", "./vectordb/evaluation_evidence")
    )
    evaluation_collection_prefix: str = os.getenv(
        "EVALUATION_COLLECTION_PREFIX", "perception_evaluations"
    )
    historical_results_k: int = int(os.getenv("HISTORICAL_RESULTS_K", "3"))

    @property
    def project1_mcp_python(self) -> Path:
        configured = os.getenv("PROJECT1_MCP_PYTHON")
        if configured:
            return Path(configured).expanduser().resolve()
        return self.project1_mcp_project_dir / ".venv" / "bin" / "python"

    @property
    def evaluation_collection_name(self) -> str:
        prefix = re.sub(
            r"[^a-z0-9_-]+", "_", self.evaluation_collection_prefix.lower()
        ).strip("_-")[:28]
        return f"{prefix}_{_collection_suffix(self.local_embedding_model)}"


settings = Settings()

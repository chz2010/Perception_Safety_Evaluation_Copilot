"""Local semantic index for Project 3 evaluation and failure evidence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .scenario_retrieval import RetrievedContext
from .settings import Settings, settings


class LocalEmbeddingUnavailable(RuntimeError):
    """Raised when the optional local embedding stack is not available."""


@dataclass(frozen=True)
class EvaluationEvidence:
    evaluation_id: int
    text: str
    metadata: dict[str, str | int | float | bool]


def build_evaluation_evidence(
    *,
    evaluation_id: int,
    scenario_name: str,
    image_name: str,
    model_name: str,
    metrics: dict[str, Any],
    report_markdown: str,
    review_notes: str = "",
) -> EvaluationEvidence:
    """Convert one saved evaluation into stable text and scalar metadata."""
    missed = metrics.get("missed_objects") or {}
    false_positives = metrics.get("false_positives") or {}
    visibility = str(metrics.get("visibility_level") or "unknown")
    severity = str(metrics.get("safety_lens_severity") or "unknown")
    text = "\n".join(
        [
            f"Evaluation ID: {evaluation_id}",
            f"Scenario: {scenario_name or 'unspecified'}",
            f"Image: {image_name}",
            f"Model: {model_name}",
            f"Severity: {severity}",
            f"Visibility: {visibility}",
            f"Missed objects: {missed}",
            f"False positives: {false_positives}",
            f"Precision: {metrics.get('precision')}",
            f"Recall: {metrics.get('recall')}",
            f"Review notes: {review_notes or 'none'}",
            "Report excerpt:",
            report_markdown[:5000],
        ]
    )
    metadata: dict[str, str | int | float | bool] = {
        "evaluation_id": evaluation_id,
        "scenario": scenario_name or "unspecified",
        "image_name": image_name,
        "model_name": model_name,
        "severity": severity,
        "visibility": visibility,
        "missed_total": int(sum(int(value) for value in missed.values())),
        "retrieval_method": "project3_local_embedding",
        "source_type": "historical_evaluation",
    }
    if metrics.get("recall") is not None:
        metadata["recall"] = float(metrics["recall"])
    return EvaluationEvidence(evaluation_id=evaluation_id, text=text, metadata=metadata)


class LocalEvaluationIndex:
    """Chroma index backed by a local SentenceTransformer embedding model.

    Optional constructor injection keeps retrieval testable without downloading a
    model or starting a real Chroma client.
    """

    def __init__(
        self,
        config: Settings = settings,
        *,
        embedder: Any | None = None,
        collection: Any | None = None,
    ) -> None:
        self.config = config
        self._embedder = embedder
        self._collection = collection

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalEmbeddingUnavailable(
                "Local embeddings require sentence-transformers. Install the project requirements."
            ) from exc
        self._embedder = SentenceTransformer(
            self.config.local_embedding_model,
            device=self.config.local_embedding_device,
        )
        return self._embedder

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        try:
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise LocalEmbeddingUnavailable(
                "Local evaluation indexing requires chromadb. Install the project requirements."
            ) from exc
        self.config.evaluation_chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.config.evaluation_chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=self.config.evaluation_collection_name,
            metadata={
                "embedding_model": self.config.local_embedding_model,
                "owner": "perception_safety_evaluation_copilot",
            },
        )
        return self._collection

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_embedder().encode(texts, normalize_embeddings=True)
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]

    def upsert(self, evidence: EvaluationEvidence) -> None:
        collection = self._get_collection()
        collection.upsert(
            ids=[f"evaluation-{evidence.evaluation_id}"],
            documents=[evidence.text],
            embeddings=self._encode([evidence.text]),
            metadatas=[evidence.metadata],
        )

    def search(self, query: str, limit: int | None = None) -> list[RetrievedContext]:
        if not query.strip():
            return []
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(
            query_embeddings=self._encode([query]),
            n_results=min(limit or self.config.historical_results_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        contexts: list[RetrievedContext] = []
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            metadata = dict(metadata or {})
            contexts.append(
                RetrievedContext(
                    evidence_id=f"HIST-{index}",
                    title=f"Historical evaluation {metadata.get('evaluation_id', index)}",
                    source_path=None,
                    layer="historical_evaluations",
                    score=round(1.0 / (1.0 + float(distance)), 4),
                    matched_terms=[],
                    excerpt=str(document)[:1200],
                    retrieval_reason="Semantically similar Project 3 evaluation evidence.",
                    source_type="historical_evaluation",
                    retrieval_method="project3_local_embedding",
                    metadata={
                        **metadata,
                        "embedding_model": self.config.local_embedding_model,
                        "distance": float(distance),
                    },
                )
            )
        return contexts

    def status(self) -> dict[str, Any]:
        if not self.config.local_embeddings_enabled:
            return {"enabled": False, "available": False, "status": "disabled"}
        try:
            count = self._get_collection().count()
        except Exception as exc:
            return {
                "enabled": True,
                "available": False,
                "status": "unavailable",
                "message": str(exc),
            }
        return {
            "enabled": True,
            "available": True,
            "status": "ok",
            "document_count": count,
            "model": self.config.local_embedding_model,
            "collection": self.config.evaluation_collection_name,
        }

"""Route Project 3 retrieval across MCP, local embeddings, and lexical fallback."""

from __future__ import annotations

from typing import Any

from .local_embeddings import LocalEmbeddingUnavailable, LocalEvaluationIndex
from .mcp_client import Project1McpClient, Project1McpUnavailable
from .scenario_retrieval import (
    RetrievalBundle,
    build_query_plan,
    retrieve_project1_evidence,
)
from .settings import Settings, settings


class RetrievalOrchestrator:
    def __init__(
        self,
        config: Settings = settings,
        *,
        mcp_client: Project1McpClient | None = None,
        evaluation_index: LocalEvaluationIndex | None = None,
    ) -> None:
        self.config = config
        self.mcp_client = mcp_client or Project1McpClient(config)
        self.evaluation_index = evaluation_index or LocalEvaluationIndex(config)

    def retrieve(
        self,
        *,
        scenario_name: str,
        scenario_tags: list[str],
        detected_objects: dict[str, int],
        expected_objects: dict[str, int],
        low_confidence_expected_objects: dict[str, int],
        missed_expected_objects: dict[str, int],
        include_project1: bool = True,
    ) -> RetrievalBundle:
        query_plan = build_query_plan(
            scenario_name,
            scenario_tags,
            detected_objects,
            expected_objects,
            low_confidence_expected_objects,
            missed_expected_objects,
        )
        query_terms = list(dict.fromkeys(term for values in query_plan.values() for term in values))
        has_failure = bool(low_confidence_expected_objects or missed_expected_objects)
        grounding_notes: list[str] = []
        metadata: dict[str, Any] = {
            "project1_method": "disabled" if not include_project1 else "not_attempted",
            "historical_method": "disabled",
            "fallback_used": False,
        }

        historical = []
        if self.config.local_embeddings_enabled and (scenario_name.strip() or has_failure):
            try:
                historical = self.evaluation_index.search(" ".join(query_terms))
                metadata["historical_method"] = "project3_local_embedding"
                metadata["historical_result_count"] = len(historical)
                metadata["embedding_model"] = self.config.local_embedding_model
            except LocalEmbeddingUnavailable as exc:
                metadata["historical_method"] = "unavailable"
                metadata["historical_error_type"] = exc.__class__.__name__
                grounding_notes.append("Local historical-evaluation embeddings were unavailable.")
            except Exception as exc:
                metadata["historical_method"] = "failed"
                metadata["historical_error_type"] = exc.__class__.__name__
                grounding_notes.append("Historical-evaluation retrieval failed safely.")

        if not include_project1:
            return RetrievalBundle(
                query_terms=query_terms,
                query_plan=query_plan,
                grounding_notes=grounding_notes + ["Project 1 retrieval was disabled for this run."],
                similar_scenarios=[],
                failure_mechanisms=[],
                safety_context=[],
                standards_guidance=[],
                historical_evaluations=historical,
                retrieval_metadata=metadata,
            )

        if self.config.project1_mcp_enabled and has_failure:
            try:
                standards, videos = self.mcp_client.retrieve(
                    query_plan=query_plan,
                    has_failure_evidence=has_failure,
                )
                metadata["project1_method"] = "project1_mcp"
                metadata["project1_standard_results"] = len(standards)
                metadata["project1_video_results"] = len(videos)
                return RetrievalBundle(
                    query_terms=query_terms,
                    query_plan=query_plan,
                    grounding_notes=grounding_notes,
                    similar_scenarios=[],
                    failure_mechanisms=videos,
                    safety_context=[],
                    standards_guidance=standards,
                    historical_evaluations=historical,
                    retrieval_metadata=metadata,
                )
            except Project1McpUnavailable as exc:
                metadata["project1_mcp_error_type"] = exc.__class__.__name__
                grounding_notes.append(
                    "Live Project 1 MCP retrieval was unavailable; the configured fallback was evaluated."
                )

        if self.config.lexical_fallback_enabled:
            fallback = retrieve_project1_evidence(
                scenario_name=scenario_name,
                scenario_tags=scenario_tags,
                detected_objects=detected_objects,
                expected_objects=expected_objects,
                low_confidence_expected_objects=low_confidence_expected_objects,
                missed_expected_objects=missed_expected_objects,
            )
            fallback.grounding_notes[:0] = grounding_notes
            fallback.historical_evaluations.extend(historical)
            fallback.retrieval_metadata.update(metadata)
            fallback.retrieval_metadata["project1_method"] = "local_lexical_fallback"
            fallback.retrieval_metadata["fallback_used"] = True
            return fallback

        grounding_notes.append("No Project 1 fallback retrieval was enabled.")
        return RetrievalBundle(
            query_terms=query_terms,
            query_plan=query_plan,
            grounding_notes=grounding_notes,
            similar_scenarios=[],
            failure_mechanisms=[],
            safety_context=[],
            standards_guidance=[],
            historical_evaluations=historical,
            retrieval_metadata=metadata,
        )

    def status(self) -> dict[str, Any]:
        return {
            "project1_mcp": self.mcp_client.status(),
            "local_embeddings": self.evaluation_index.status(),
        }

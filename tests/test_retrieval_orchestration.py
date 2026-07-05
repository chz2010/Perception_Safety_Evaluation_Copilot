from dataclasses import replace

from src.perception_safety_copilot.local_embeddings import (
    LocalEvaluationIndex,
    build_evaluation_evidence,
)
from src.perception_safety_copilot.mcp_client import (
    Project1McpClient,
    Project1McpUnavailable,
)
from src.perception_safety_copilot.retrieval_orchestrator import RetrievalOrchestrator
from src.perception_safety_copilot.scenario_retrieval import RetrievedContext
from src.perception_safety_copilot.settings import settings


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query(self, **kwargs):
        return {
            "documents": [["Scenario: nighttime crosswalk\nMissed objects: {'person': 1}"]],
            "metadatas": [[{"evaluation_id": 7, "severity": "CRITICAL"}]],
            "distances": [[0.25]],
        }

    def count(self):
        return max(1, len(self.upserts))


def test_local_evaluation_index_upserts_and_retrieves_with_provenance():
    collection = FakeCollection()
    index = LocalEvaluationIndex(embedder=FakeEmbedder(), collection=collection)
    evidence = build_evaluation_evidence(
        evaluation_id=7,
        scenario_name="Nighttime crosswalk",
        image_name="scene.jpg",
        model_name="yolo11m",
        metrics={
            "missed_objects": {"person": 1},
            "false_positives": {},
            "recall": 0.0,
            "visibility_level": "low",
            "safety_lens_severity": "CRITICAL",
        },
        report_markdown="# Report",
    )

    index.upsert(evidence)
    results = index.search("missed pedestrian at night")

    assert collection.upserts[0]["ids"] == ["evaluation-7"]
    assert results[0].retrieval_method == "project3_local_embedding"
    assert results[0].source_type == "historical_evaluation"
    assert results[0].metadata["evaluation_id"] == 7


class FakeMcpClient(Project1McpClient):
    def call_tools(self, requests):
        payloads = []
        for tool_name, arguments in requests:
            if tool_name == "search_safety_standards":
                payloads.append(
                    {
                        "status": "ok",
                        "results": [
                            {
                                "rank": 1,
                                "source_type": "standards",
                                "content": f"Evidence for {arguments['standard']}",
                                "metadata": {"standard": arguments["standard"], "page": 4},
                            }
                        ],
                    }
                )
            else:
                payloads.append(
                    {
                        "status": "ok",
                        "results": [
                            {
                                "rank": 1,
                                "source_type": "video_transcript",
                                "content": "Night pedestrian failure evidence",
                                "metadata": {"title": "Failure video"},
                            }
                        ],
                    }
                )
        return payloads


def test_mcp_client_maps_standards_and_video_results():
    client = FakeMcpClient(replace(settings, project1_mcp_video_enabled=True))
    plan = {
        "failure_mechanism": ["missed detection", "pedestrian"],
        "sotif": ["triggering condition"],
        "iso_8800": ["model performance"],
        "iso_26262": ["fallback"],
    }

    standards, videos = client.retrieve(query_plan=plan, has_failure_evidence=True)

    assert len(standards) == 3
    assert len(videos) == 1
    assert all(item.retrieval_method == "project1_mcp" for item in standards + videos)
    assert {item.metadata.get("standard") for item in standards} == {
        "ISO 21448",
        "ISO 8800",
        "ISO 26262",
    }


class FailingMcpClient:
    def retrieve(self, **kwargs):
        raise Project1McpUnavailable("controlled test failure")

    def status(self):
        return {"status": "unavailable"}


class FakeHistoricalIndex:
    def search(self, query):
        return [
            RetrievedContext(
                evidence_id="HIST-1",
                title="Historical evaluation 1",
                source_path=None,
                layer="historical_evaluations",
                score=0.9,
                matched_terms=[],
                excerpt="Earlier missed pedestrian evaluation",
                retrieval_reason="Semantic match",
                source_type="historical_evaluation",
                retrieval_method="project3_local_embedding",
            )
        ]

    def status(self):
        return {"status": "ok"}


def test_orchestrator_uses_explicit_lexical_fallback_when_mcp_fails():
    config = replace(
        settings,
        project1_mcp_enabled=True,
        lexical_fallback_enabled=True,
        local_embeddings_enabled=True,
    )
    orchestrator = RetrievalOrchestrator(
        config,
        mcp_client=FailingMcpClient(),
        evaluation_index=FakeHistoricalIndex(),
    )

    bundle = orchestrator.retrieve(
        scenario_name="Nighttime urban crosswalk",
        scenario_tags=["night", "urban", "crosswalk"],
        detected_objects={},
        expected_objects={"person": 1},
        low_confidence_expected_objects={},
        missed_expected_objects={"person": 1},
    )

    assert bundle.retrieval_metadata["project1_method"] == "local_lexical_fallback"
    assert bundle.retrieval_metadata["fallback_used"] is True
    assert bundle.historical_evaluations[0].evidence_id == "HIST-1"
    assert bundle.standards_guidance
    assert any("MCP retrieval was unavailable" in note for note in bundle.grounding_notes)


def test_orchestrator_does_not_force_project1_evidence_without_failure():
    config = replace(settings, local_embeddings_enabled=False)
    orchestrator = RetrievalOrchestrator(
        config,
        mcp_client=FailingMcpClient(),
        evaluation_index=FakeHistoricalIndex(),
    )
    bundle = orchestrator.retrieve(
        scenario_name="",
        scenario_tags=[],
        detected_objects={},
        expected_objects={},
        low_confidence_expected_objects={},
        missed_expected_objects={},
    )

    assert bundle.standards_guidance == []
    assert bundle.failure_mechanisms == []
    assert any("skipped" in note.lower() for note in bundle.grounding_notes)

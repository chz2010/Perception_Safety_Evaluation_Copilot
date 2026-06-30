from src.perception_safety_copilot.scenario_retrieval import (
    build_query_plan,
    retrieve_project1_evidence,
)
from src.perception_safety_copilot.safety_lens import evaluate_safety_lens, infer_scenario_tags


def _sample_safety_result():
    return evaluate_safety_lens(
        raw_detections=[],
        display_detections=[],
        expected_objects={"person": 1, "traffic light": 1},
        display_threshold=0.80,
        low_conf_threshold=0.50,
        scenario_tags=infer_scenario_tags("Nighttime urban crosswalk with glare"),
        metrics={"recall": 0.0},
    )


def test_retrieve_project1_evidence_returns_guidance():
    safety_result = _sample_safety_result()
    bundle = retrieve_project1_evidence(
        scenario_name="Nighttime urban crosswalk with glare",
        scenario_tags=safety_result.scenario_tags,
        expected_objects=safety_result.expected_objects,
        low_confidence_expected_objects=safety_result.low_confidence_expected_objects,
        missed_expected_objects=safety_result.missed_expected_objects,
    )

    assert bundle.standards_guidance
    assert bundle.failure_mechanisms
    assert any("ISO" in item.title or "SOTIF" in item.title for item in bundle.standards_guidance)
    assert all("Lane Maintaining" not in item.title for item in bundle.similar_scenarios)
    assert all("LiDAR" not in item.title for item in bundle.failure_mechanisms)
    assert all(item.evidence_id for item in bundle.standards_guidance)
    assert {item.evidence_id for item in bundle.standards_guidance} == {
        "STD-SOTIF-1",
        "STD-8800-1",
        "STD-26262-1",
    }


def test_multi_retrieval_builds_distinct_queries():
    query_plan = build_query_plan(
        scenario_name="Rainy nighttime crosswalk",
        scenario_tags=["rain", "night", "crosswalk"],
        detected_objects={"car": 2},
        expected_objects={"person": 1},
        low_confidence_expected_objects={"person": 1},
        missed_expected_objects={},
    )

    assert "low light" in query_plan["scenario_similarity"]
    assert "calibration" in query_plan["failure_mechanism"]
    assert "triggering condition" in query_plan["sotif"]
    assert "data quality" in query_plan["iso_8800"]
    assert "controllability" in query_plan["iso_26262"]


def test_retrieval_does_not_force_scenario_without_scene_context():
    safety_result = evaluate_safety_lens(
        raw_detections=[],
        display_detections=[],
        expected_objects={},
        display_threshold=0.80,
        low_conf_threshold=0.50,
        scenario_tags=[],
    )
    bundle = retrieve_project1_evidence(
        scenario_name="",
        scenario_tags=[],
        detected_objects=safety_result.detected_objects,
        expected_objects=safety_result.expected_objects,
        low_confidence_expected_objects=safety_result.low_confidence_expected_objects,
        missed_expected_objects=safety_result.missed_expected_objects,
    )

    assert bundle.similar_scenarios == []
    assert bundle.failure_mechanisms == []
    assert bundle.standards_guidance == []
    assert any("skipped" in note.lower() for note in bundle.grounding_notes)

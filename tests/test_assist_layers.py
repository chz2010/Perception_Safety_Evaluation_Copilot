from src.perception_safety_copilot.llm_assist import build_llm_assist_payload
from src.perception_safety_copilot.scenario_retrieval import retrieve_project1_evidence
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
    assert any("ISO" in item.title or "SOTIF" in item.title for item in bundle.standards_guidance)


def test_build_llm_assist_payload_contains_layered_structure():
    safety_result = _sample_safety_result()
    bundle = retrieve_project1_evidence(
        scenario_name="Nighttime urban crosswalk with glare",
        scenario_tags=safety_result.scenario_tags,
        expected_objects=safety_result.expected_objects,
        low_confidence_expected_objects=safety_result.low_confidence_expected_objects,
        missed_expected_objects=safety_result.missed_expected_objects,
    )

    payload = build_llm_assist_payload(
        scenario_name="Nighttime urban crosswalk with glare",
        scenario_tags=safety_result.scenario_tags,
        safety_result=safety_result,
        retrieval_bundle=bundle,
        metrics={"precision": 1.0, "recall": 0.17, "map50": None, "map50_95": None},
    )

    assert "deterministic_layer" in payload
    assert "scenario_retrieval_layer" in payload
    assert payload["deterministic_layer"]["severity"] == safety_result.severity
    assert "similar_known_scenarios" in payload["scenario_retrieval_layer"]

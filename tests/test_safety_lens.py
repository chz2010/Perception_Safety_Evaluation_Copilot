from src.perception_safety_copilot.evaluation import Detection
from src.perception_safety_copilot.safety_lens import (
    evaluate_safety_lens,
    generate_safety_report,
    infer_scenario_tags,
    parse_expected_objects,
)


def test_parse_expected_objects_accepts_common_formats():
    parsed = parse_expected_objects("person: 1\ncar,2\npedestrian\n# ignore")

    assert parsed == {"person": 1, "car": 2, "pedestrian": 1}


def test_expected_vru_below_display_threshold_is_high_not_missed():
    raw_detections = [
        Detection("person", 0.51, (0, 0, 10, 10), image_id="img-1", model_name="yolo11s.pt"),
    ]
    display_detections = []

    result = evaluate_safety_lens(
        raw_detections=raw_detections,
        display_detections=display_detections,
        expected_objects={"person": 1},
        display_threshold=0.90,
        low_conf_threshold=0.50,
        scenario_tags=infer_scenario_tags("Nighttime urban crosswalk"),
        metrics={"recall": 0.0},
    )

    assert result.severity == "HIGH"
    assert result.low_confidence_expected_objects == {"person": 1}
    assert result.missed_expected_objects == {}
    assert any("SOTIF" in finding.standard for finding in result.standard_specific_findings)
    assert any("ISO 8800" in finding.standard for finding in result.standard_specific_findings)


def test_expected_vru_missing_from_raw_is_critical():
    result = evaluate_safety_lens(
        raw_detections=[],
        display_detections=[],
        expected_objects={"person": 1},
        display_threshold=0.90,
        low_conf_threshold=0.50,
        scenario_tags=infer_scenario_tags("Nighttime urban crosswalk"),
    )

    assert result.severity == "CRITICAL"
    assert result.low_confidence_expected_objects == {}
    assert result.missed_expected_objects == {"person": 1}
    assert any("insufficient" in finding.interpretation.lower() or "relevant" in finding.interpretation.lower() for finding in result.standard_specific_findings)


def test_generate_safety_report_includes_required_sections():
    result = evaluate_safety_lens(
        raw_detections=[],
        display_detections=[],
        expected_objects={"person": 1},
        display_threshold=0.90,
        low_conf_threshold=0.50,
        scenario_tags=infer_scenario_tags("Nighttime urban crosswalk"),
        metrics={"recall": 0.0},
    )

    report = generate_safety_report(
        result,
        scenario_name="Nighttime urban crosswalk",
        metrics={
            "precision": 1.0,
            "recall": 0.17,
            "display_threshold": 0.80,
            "low_confidence_threshold": 0.50,
            "map50": None,
            "map50_95": None,
            "ground_truth_boxes_available": False,
        },
    )

    assert "## Safety Lens Assessment" in report
    assert "Severity: CRITICAL" in report
    assert "Primary safety concern:" in report
    assert "Observed perception issues:" in report
    assert "ISO 21448 / SOTIF:" in report
    assert "Recommended SOTIF actions:" in report
    assert "ISO 8800:" in report
    assert "Recommended ISO 8800-style actions:" in report
    assert "ISO 26262:" in report
    assert "Recommended ISO 26262-style actions:" in report
    assert "Evidence summary:" in report
    assert "Final conclusion:" in report

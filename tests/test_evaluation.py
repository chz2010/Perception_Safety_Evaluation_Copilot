from src.perception_safety_copilot.evaluation import (
    Detection,
    evaluate_detections,
    parse_ground_truth,
)


def test_parse_ground_truth_accepts_common_formats():
    expected = parse_ground_truth("person: 2\ncar,1\ntraffic light\n# ignored")

    assert expected == {"person": 2, "car": 1, "traffic light": 1}


def test_evaluate_detections_with_ground_truth_counts():
    detections = [
        Detection("person", 0.91, (0, 0, 10, 10)),
        Detection("car", 0.42, (0, 0, 10, 10)),
        Detection("truck", 0.88, (0, 0, 10, 10)),
    ]

    result = evaluate_detections(
        detections,
        expected_counts={"person": 1, "car": 2},
        low_confidence_threshold=0.5,
    )

    assert result.detected_counts == {"person": 1, "car": 1, "truck": 1}
    assert result.missed_objects == {"car": 1}
    assert result.false_positives == {"truck": 1}
    assert len(result.low_confidence_detections) == 1
    assert round(result.precision, 2) == 0.67
    assert round(result.recall, 2) == 0.67


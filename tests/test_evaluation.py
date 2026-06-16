from src.perception_safety_copilot.evaluation import (
    Detection,
    GroundTruthBox,
    box_iou,
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


def test_box_iou_calculates_intersection_over_union():
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_evaluate_detections_with_iou_and_map_metrics():
    detections = [
        Detection("car", 0.95, (0, 0, 10, 10)),
        Detection("car", 0.80, (40, 40, 50, 50)),
        Detection("person", 0.70, (20, 20, 30, 30)),
    ]
    ground_truth_boxes = [
        GroundTruthBox("car", (0, 0, 10, 10)),
        GroundTruthBox("person", (20, 20, 30, 30)),
    ]

    result = evaluate_detections(
        detections,
        ground_truth_boxes=ground_truth_boxes,
        low_confidence_threshold=0.5,
    )

    assert result.iou_evaluation is not None
    assert result.iou_evaluation.matched_detections == 2
    assert result.iou_evaluation.unmatched_detections == 1
    assert result.iou_evaluation.unmatched_ground_truth == 0
    assert result.iou_evaluation.precision == 2 / 3
    assert result.iou_evaluation.recall == 1.0
    assert result.iou_evaluation.mean_iou == 1.0
    assert result.map50 == 1.0
    assert result.map50_95 == 1.0


def test_map50_95_penalizes_loose_boxes_more_than_map50():
    detections = [
        Detection("car", 0.95, (0, 0, 10, 10)),
        Detection("person", 0.90, (20, 20, 32, 32)),
    ]
    ground_truth_boxes = [
        GroundTruthBox("car", (0, 0, 10, 10)),
        GroundTruthBox("person", (20, 20, 30, 30)),
    ]

    result = evaluate_detections(detections, ground_truth_boxes=ground_truth_boxes)

    assert result.map50 == 1.0
    assert result.map50_95 is not None
    assert result.map50_95 < result.map50

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    image_id: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class GroundTruthBox:
    label: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class IoUEvaluation:
    threshold: float
    matched_detections: int
    unmatched_detections: int
    unmatched_ground_truth: int
    precision: float | None
    recall: float | None
    mean_iou: float | None


@dataclass(frozen=True)
class EvaluationResult:
    detected_counts: dict[str, int]
    expected_counts: dict[str, int]
    missed_objects: dict[str, int]
    false_positives: dict[str, int]
    low_confidence_detections: list[Detection]
    precision: float | None
    recall: float | None
    iou_evaluation: IoUEvaluation | None = None
    map50: float | None = None
    map50_95: float | None = None


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def parse_ground_truth(text: str) -> dict[str, int]:
    """Parse expected objects from lines like 'pedestrian: 2' or 'car,1'."""
    expected: Counter[str] = Counter()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            label, count = line.split(":", 1)
        elif "," in line:
            label, count = line.split(",", 1)
        else:
            label, count = line, "1"

        label = normalize_label(label)
        if not label:
            continue

        try:
            parsed_count = int(count.strip())
        except ValueError:
            parsed_count = 1

        expected[label] += max(parsed_count, 0)

    return dict(expected)


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _interpolated_average_precision(
    true_positives: list[int],
    false_positives: list[int],
    total_ground_truth: int,
) -> float | None:
    if total_ground_truth == 0:
        return None

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = [0.0]
    precisions = [1.0]

    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / total_ground_truth)
        denominator = cumulative_tp + cumulative_fp
        precisions.append(cumulative_tp / denominator if denominator else 0.0)

    recalls.append(1.0)
    precisions.append(0.0)

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    average_precision = 0.0
    for index in range(1, len(recalls)):
        if recalls[index] > recalls[index - 1]:
            average_precision += (recalls[index] - recalls[index - 1]) * precisions[index]
    return average_precision


def calculate_average_precision(
    detections: Iterable[Detection],
    ground_truth_boxes: Iterable[GroundTruthBox],
    iou_threshold: float,
) -> float | None:
    detections_by_label: dict[str, list[Detection]] = {}
    ground_truth_by_label: dict[str, list[GroundTruthBox]] = {}

    for detection in detections:
        detections_by_label.setdefault(normalize_label(detection.label), []).append(detection)
    for box in ground_truth_boxes:
        ground_truth_by_label.setdefault(normalize_label(box.label), []).append(box)

    ap_values: list[float] = []
    for label, label_ground_truth in ground_truth_by_label.items():
        label_detections = sorted(
            detections_by_label.get(label, []),
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        matched_ground_truth: set[int] = set()
        true_positives: list[int] = []
        false_positives: list[int] = []

        for detection in label_detections:
            best_index = None
            best_iou = 0.0
            for index, ground_truth in enumerate(label_ground_truth):
                if index in matched_ground_truth:
                    continue
                iou = box_iou(detection.bbox_xyxy, ground_truth.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None and best_iou >= iou_threshold:
                matched_ground_truth.add(best_index)
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        ap = _interpolated_average_precision(
            true_positives,
            false_positives,
            total_ground_truth=len(label_ground_truth),
        )
        if ap is not None:
            ap_values.append(ap)

    if not ap_values:
        return None
    return sum(ap_values) / len(ap_values)


def evaluate_iou_matches(
    detections: Iterable[Detection],
    ground_truth_boxes: Iterable[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> IoUEvaluation | None:
    detections = sorted(list(detections), key=lambda detection: detection.confidence, reverse=True)
    ground_truth_boxes = list(ground_truth_boxes)
    if not ground_truth_boxes:
        return None

    matched_ground_truth: set[int] = set()
    matched_ious: list[float] = []
    unmatched_detections = 0

    for detection in detections:
        detection_label = normalize_label(detection.label)
        best_index = None
        best_iou = 0.0
        for index, ground_truth in enumerate(ground_truth_boxes):
            if index in matched_ground_truth:
                continue
            if normalize_label(ground_truth.label) != detection_label:
                continue
            iou = box_iou(detection.bbox_xyxy, ground_truth.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_ground_truth.add(best_index)
            matched_ious.append(best_iou)
        else:
            unmatched_detections += 1

    matched_count = len(matched_ground_truth)
    unmatched_ground_truth = len(ground_truth_boxes) - matched_count
    precision = matched_count / len(detections) if detections else 0.0
    recall = matched_count / len(ground_truth_boxes)

    return IoUEvaluation(
        threshold=iou_threshold,
        matched_detections=matched_count,
        unmatched_detections=unmatched_detections,
        unmatched_ground_truth=unmatched_ground_truth,
        precision=precision,
        recall=recall,
        mean_iou=sum(matched_ious) / len(matched_ious) if matched_ious else None,
    )


def evaluate_detections(
    detections: Iterable[Detection],
    expected_counts: dict[str, int] | None = None,
    low_confidence_threshold: float = 0.5,
    ground_truth_boxes: Iterable[GroundTruthBox] | None = None,
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    detections = list(detections)
    ground_truth_boxes = list(ground_truth_boxes or [])
    detected_counts = Counter(normalize_label(d.label) for d in detections)
    expected = Counter({normalize_label(k): int(v) for k, v in (expected_counts or {}).items()})
    if ground_truth_boxes and not expected:
        expected = Counter(normalize_label(box.label) for box in ground_truth_boxes)

    missed = Counter()
    false_positives = Counter()
    true_positive_count = 0

    all_labels = set(detected_counts) | set(expected)
    for label in all_labels:
        detected = detected_counts[label]
        wanted = expected[label]
        true_positive_count += min(detected, wanted)
        if wanted > detected:
            missed[label] = wanted - detected
        if detected > wanted:
            false_positives[label] = detected - wanted

    low_confidence = [d for d in detections if d.confidence < low_confidence_threshold]

    if expected:
        total_detections = sum(detected_counts.values())
        total_expected = sum(expected.values())
        precision = true_positive_count / total_detections if total_detections else 0.0
        recall = true_positive_count / total_expected if total_expected else 0.0
    else:
        precision = None
        recall = None
        false_positives.clear()

    iou_evaluation = evaluate_iou_matches(detections, ground_truth_boxes, iou_threshold)
    map50 = calculate_average_precision(detections, ground_truth_boxes, 0.5)
    map_thresholds = [threshold / 100 for threshold in range(50, 100, 5)]
    ap_values = [
        ap
        for threshold in map_thresholds
        if (ap := calculate_average_precision(detections, ground_truth_boxes, threshold)) is not None
    ]
    map50_95 = sum(ap_values) / len(ap_values) if ap_values else None

    return EvaluationResult(
        detected_counts=dict(detected_counts),
        expected_counts=dict(expected),
        missed_objects=dict(missed),
        false_positives=dict(false_positives),
        low_confidence_detections=low_confidence,
        precision=precision,
        recall=recall,
        iou_evaluation=iou_evaluation,
        map50=map50,
        map50_95=map50_95,
    )

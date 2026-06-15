from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class EvaluationResult:
    detected_counts: dict[str, int]
    expected_counts: dict[str, int]
    missed_objects: dict[str, int]
    false_positives: dict[str, int]
    low_confidence_detections: list[Detection]
    precision: float | None
    recall: float | None


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


def evaluate_detections(
    detections: Iterable[Detection],
    expected_counts: dict[str, int] | None = None,
    low_confidence_threshold: float = 0.5,
) -> EvaluationResult:
    detections = list(detections)
    detected_counts = Counter(normalize_label(d.label) for d in detections)
    expected = Counter({normalize_label(k): int(v) for k, v in (expected_counts or {}).items()})

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

    return EvaluationResult(
        detected_counts=dict(detected_counts),
        expected_counts=dict(expected),
        missed_objects=dict(missed),
        false_positives=dict(false_positives),
        low_confidence_detections=low_confidence,
        precision=precision,
        recall=recall,
    )


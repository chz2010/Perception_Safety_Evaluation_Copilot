from __future__ import annotations

from datetime import datetime

from .evaluation import Detection, EvaluationResult


SAFETY_RELEVANT_CLASSES = {
    "person": "vulnerable road user",
    "pedestrian": "vulnerable road user",
    "bicycle": "vulnerable road user",
    "motorcycle": "vulnerable road user",
    "car": "traffic participant",
    "bus": "traffic participant",
    "truck": "traffic participant",
    "traffic light": "traffic control",
    "stop sign": "traffic control",
}


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "None"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def describe_safety_impact(result: EvaluationResult) -> list[str]:
    impacts: list[str] = []

    for label, count in sorted(result.missed_objects.items()):
        role = SAFETY_RELEVANT_CLASSES.get(label, "safety-relevant object")
        impacts.append(
            f"Missed {label} ({count}) may indicate insufficient perception coverage for a {role}."
        )

    for detection in result.low_confidence_detections:
        role = SAFETY_RELEVANT_CLASSES.get(detection.label, "object")
        impacts.append(
            f"Low confidence for {detection.label} ({detection.confidence:.2f}) may reduce downstream planning margin for this {role}."
        )

    if result.false_positives:
        impacts.append(
            "False positives may trigger unnecessary braking, evasive maneuvers, or driver alerts."
        )

    if not impacts:
        impacts.append(
            "No obvious perception failure was identified in this MVP evaluation. Confirm with broader scenes and labeled data."
        )

    return impacts


def recommended_tests(result: EvaluationResult) -> list[str]:
    tests = [
        "Repeat the scenario with multiple confidence thresholds and document threshold sensitivity.",
    ]
    if result.iou_evaluation is None:
        tests.append("Add labeled ground truth boxes so future evaluation can use IoU-based object matching.")
    else:
        tests.append("Review unmatched IoU detections and missed boxes as formal perception failure candidates.")

    if result.missed_objects:
        tests.append("Create focused regression tests for each missed object class and similar edge cases.")
    if result.low_confidence_detections:
        tests.append("Add lighting, occlusion, distance, and weather variants for low-confidence classes.")
    if result.false_positives:
        tests.append("Review false-positive classes against background clutter and unusual road geometry.")

    return tests


def generate_markdown_report(
    scenario_name: str,
    image_name: str,
    detections: list[Detection],
    result: EvaluationResult,
    confidence_threshold: float,
    low_confidence_threshold: float,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = [
        f"- Detected objects: {len(detections)}",
        f"- Detection threshold: {confidence_threshold:.2f}",
        f"- Low-confidence threshold: {low_confidence_threshold:.2f}",
    ]
    if result.precision is not None and result.recall is not None:
        metrics.extend(
            [
                f"- Precision: {result.precision:.2f}",
                f"- Recall: {result.recall:.2f}",
            ]
        )
    else:
        metrics.append("- Precision/recall: not available without ground truth")
    if result.iou_evaluation is not None:
        iou = result.iou_evaluation
        metrics.extend(
            [
                f"- IoU threshold: {iou.threshold:.2f}",
                f"- IoU matched detections: {iou.matched_detections}",
                f"- IoU unmatched detections: {iou.unmatched_detections}",
                f"- IoU unmatched ground truth: {iou.unmatched_ground_truth}",
                f"- Mean matched IoU: {'N/A' if iou.mean_iou is None else f'{iou.mean_iou:.3f}'}",
                f"- mAP50: {'N/A' if result.map50 is None else f'{result.map50:.3f}'}",
                f"- mAP50-95: {'N/A' if result.map50_95 is None else f'{result.map50_95:.3f}'}",
            ]
        )
    else:
        metrics.extend(
            [
                "- IoU threshold: 0.50 (not applied without bounding-box ground truth)",
                "- mAP50: not available without bounding-box ground truth",
                "- mAP50-95: not available without bounding-box ground truth",
            ]
        )

    lines = [
        "# Perception Safety Evaluation Report",
        "",
        "## Scenario Summary",
        f"- Scenario: {scenario_name or 'Unspecified driving scene'}",
        f"- Image: {image_name}",
        f"- Generated: {generated_at}",
        "",
        "## Metrics",
        *metrics,
        "",
        "## Object Summary",
        f"- Detected objects: {format_counts(result.detected_counts)}",
        f"- Expected objects: {format_counts(result.expected_counts)}",
        f"- Missed expected objects: {format_counts(result.missed_objects)}",
        f"- False positives: {format_counts(result.false_positives)}",
        f"- Low-confidence detections: {len(result.low_confidence_detections)}",
        "",
        "## Potential Safety Impact",
        *[f"- {impact}" for impact in describe_safety_impact(result)],
        "",
        "## Recommended Follow-up Tests",
        *[f"- {test}" for test in recommended_tests(result)],
        "",
        "## Notes",
        "- Count-based precision/recall is used when only class counts are available.",
        "- IoU, mAP50, and mAP50-95 require bounding-box ground truth, such as projected nuScenes annotations.",
        "- Connect Project 1 for standards context and Project 2 for requirements, traceability, test cases, AgentOps, and MLflow workflows.",
    ]
    return "\n".join(lines)

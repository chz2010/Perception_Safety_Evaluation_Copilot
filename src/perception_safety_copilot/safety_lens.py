from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .evaluation import Detection


VULNERABLE_ROAD_USERS = {"person", "pedestrian", "cyclist", "bicycle", "motorcycle"}
IMPORTANT_OBJECTS = {"car", "truck", "bus", "traffic light", "stop sign"}
SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
SCENARIO_KEYWORDS = {
    "night": ["night", "nighttime", "low-light", "dark"],
    "urban": ["urban", "city", "downtown"],
    "crosswalk": ["crosswalk", "pedestrian crossing", "zebra"],
    "rain": ["rain", "wet"],
    "fog": ["fog", "mist", "haze"],
    "glare": ["glare", "sun", "headlight"],
    "occlusion": ["occlusion", "occluded", "partially hidden"],
    "intersection": ["intersection", "junction"],
    "traffic_light": ["traffic light", "signal"],
}


@dataclass(frozen=True)
class SafetyLensInput:
    detected_objects: dict[str, int]
    missed_objects: dict[str, int]
    low_confidence_expected_objects: dict[str, int]
    expected_objects: dict[str, int]
    confidence_scores: dict[str, list[float]]
    scenario_tags: list[str]
    display_threshold: float
    low_conf_threshold: float


@dataclass(frozen=True)
class StandardFinding:
    standard: str
    severity: str
    interpretation: str
    recommendations: list[str]
    evidence_chain: list[str]


@dataclass(frozen=True)
class SafetyLensV2Result:
    severity: str
    expected_objects: dict[str, int]
    detected_objects: dict[str, int]
    low_confidence_expected_objects: dict[str, int]
    missed_expected_objects: dict[str, int]
    scenario_tags: list[str]
    primary_safety_concern: str
    observed_perception_issues: list[str]
    standard_specific_findings: list[StandardFinding]
    standard_specific_recommendations: list[str]
    evidence_chain: list[str]


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def parse_expected_objects(text_input: str) -> dict[str, int]:
    expected: Counter[str] = Counter()
    for raw_line in text_input.splitlines():
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


def infer_scenario_tags(scenario_text: str) -> list[str]:
    normalized = normalize_label(scenario_text)
    tags: list[str] = []
    for tag, keywords in SCENARIO_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            tags.append(tag)
    return tags


def _count_by_label(detections: Iterable[Detection]) -> dict[str, int]:
    return dict(Counter(normalize_label(detection.label) for detection in detections))


def _group_confidences(detections: Iterable[Detection]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for detection in detections:
        grouped.setdefault(normalize_label(detection.label), []).append(float(detection.confidence))
    for values in grouped.values():
        values.sort(reverse=True)
    return grouped


def _highest_severity(findings: list[StandardFinding]) -> str:
    if not findings:
        return "LOW"
    return min(findings, key=lambda finding: SEVERITY_RANK.get(finding.severity, 99)).severity


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "None"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def _build_input(
    raw_detections: list[Detection],
    display_detections: list[Detection],
    expected_objects: dict[str, int],
    display_threshold: float,
    low_conf_threshold: float,
    scenario_tags: list[str] | None = None,
) -> SafetyLensInput:
    raw_counts = Counter(_count_by_label(raw_detections))
    display_counts = Counter(_count_by_label(display_detections))
    expected_counts = Counter({normalize_label(label): int(count) for label, count in expected_objects.items()})

    low_confidence_expected_objects: Counter[str] = Counter()
    missed_expected_objects: Counter[str] = Counter()

    for label, expected_count in expected_counts.items():
        if expected_count <= 0:
            continue

        display_count = display_counts[label]
        raw_count = raw_counts[label]
        missing_after_display = max(expected_count - display_count, 0)
        if missing_after_display <= 0:
            continue

        low_conf_count = min(max(raw_count - display_count, 0), missing_after_display)
        missed_count = max(missing_after_display - low_conf_count, 0)

        if low_conf_count > 0:
            low_confidence_expected_objects[label] += low_conf_count
        if missed_count > 0:
            missed_expected_objects[label] += missed_count

    return SafetyLensInput(
        detected_objects=dict(display_counts),
        missed_objects=dict(missed_expected_objects),
        low_confidence_expected_objects=dict(low_confidence_expected_objects),
        expected_objects=dict(expected_counts),
        confidence_scores=_group_confidences(raw_detections),
        scenario_tags=sorted(set(scenario_tags or [])),
        display_threshold=display_threshold,
        low_conf_threshold=low_conf_threshold,
    )


def _build_primary_concern(input_data: SafetyLensInput) -> str:
    scenario_parts = []
    if "night" in input_data.scenario_tags:
        scenario_parts.append("nighttime")
    if "urban" in input_data.scenario_tags:
        scenario_parts.append("urban")
    if "crosswalk" in input_data.scenario_tags:
        scenario_parts.append("crosswalk")

    context = " ".join(scenario_parts).strip()
    if any(label in VULNERABLE_ROAD_USERS for label in input_data.missed_objects | input_data.low_confidence_expected_objects):
        if context:
            return f"Potential perception limitation in a {context} scenario involving vulnerable road users."
        return "Potential perception limitation involving vulnerable road users."
    if input_data.missed_objects or input_data.low_confidence_expected_objects:
        if context:
            return f"Potential perception limitation in a {context} scenario affecting safety-relevant objects."
        return "Potential perception limitation affecting safety-relevant objects."
    return "No major expected-object issue is visible at the selected operating threshold."


def _build_observed_issues(input_data: SafetyLensInput, metrics: dict | None = None) -> list[str]:
    issues: list[str] = []

    for label, expected_count in sorted(input_data.expected_objects.items()):
        detected_count = input_data.detected_objects.get(label, 0)
        low_conf_count = input_data.low_confidence_expected_objects.get(label, 0)
        missed_count = input_data.missed_objects.get(label, 0)

        if detected_count < expected_count and detected_count > 0:
            issues.append(
                f"Expected {label} count was only partially detected above the operating threshold."
            )
        if low_conf_count > 0:
            noun = "candidate" if low_conf_count == 1 else "candidates"
            issues.append(
                f"{low_conf_count} expected {label} {'was' if low_conf_count == 1 else 'were'} detected only as low-confidence {noun}."
            )
        if missed_count > 0:
            issues.append(
                f"{missed_count} expected {label} {'was' if missed_count == 1 else 'were'} not detected."
            )

    recall = None if metrics is None else metrics.get("recall")
    if recall is not None and recall < 0.5:
        issues.append("Recall is low for the selected scenario, indicating incomplete object coverage.")

    if not issues:
        issues.append("No expected-object miss or major low-confidence issue was found for this scene.")

    return issues


def _infer_sotif(input_data: SafetyLensInput) -> StandardFinding:
    conditions = []
    if "night" in input_data.scenario_tags:
        conditions.append("nighttime lighting")
    if "crosswalk" in input_data.scenario_tags:
        conditions.append("crosswalk geometry")
    if "occlusion" in input_data.scenario_tags:
        conditions.append("partial occlusion")
    if "glare" in input_data.scenario_tags:
        conditions.append("glare")
    if "fog" in input_data.scenario_tags or "rain" in input_data.scenario_tags:
        conditions.append("low visibility")

    condition_text = ", ".join(conditions) if conditions else "specific triggering conditions"

    severity = "LOW"
    if any(label in VULNERABLE_ROAD_USERS for label in input_data.missed_objects):
        severity = "CRITICAL"
    elif any(label in VULNERABLE_ROAD_USERS for label in input_data.low_confidence_expected_objects):
        severity = "HIGH"
    elif input_data.missed_objects or input_data.low_confidence_expected_objects:
        severity = "MEDIUM"

    return StandardFinding(
        standard="ISO 21448 / SOTIF",
        severity=severity,
        interpretation=(
            "This issue is most relevant to SOTIF because the perception model may be behaving as intended, "
            f"but its performance is insufficient under {condition_text}."
        ),
        recommendations=[
            "Classify this as a potential triggering condition.",
            "Add more scenario-matched cases to the validation set.",
            "Evaluate performance across illumination, distance, pose, and occlusion variations.",
            "Compare model behavior across different confidence thresholds.",
            "Create regression tests for similar vulnerable-road-user scenarios.",
        ],
        evidence_chain=[
            f"Expected objects: {_format_counts(input_data.expected_objects)}",
            f"Detected above display threshold: {_format_counts(input_data.detected_objects)}",
            f"Low-confidence expected objects: {_format_counts(input_data.low_confidence_expected_objects)}",
            f"Missed expected objects: {_format_counts(input_data.missed_objects)}",
            f"Display threshold: {input_data.display_threshold:.2f}",
        ],
    )


def _infer_iso_8800(input_data: SafetyLensInput) -> StandardFinding:
    severity = "LOW"
    if any(label in VULNERABLE_ROAD_USERS for label in input_data.missed_objects):
        severity = "CRITICAL"
    elif any(label in VULNERABLE_ROAD_USERS for label in input_data.low_confidence_expected_objects):
        severity = "HIGH"
    elif input_data.missed_objects or input_data.low_confidence_expected_objects:
        severity = "MEDIUM"

    return StandardFinding(
        standard="ISO 8800",
        severity=severity,
        interpretation=(
            "This issue is relevant to AI safety because low-confidence and missed detections may indicate "
            "insufficient data coverage, weak confidence calibration, or limited robustness of the perception model "
            "for this scenario type."
        ),
        recommendations=[
            "Review dataset coverage for the affected object classes and scenario type.",
            "Analyze confidence calibration for vulnerable road users and safety-relevant objects.",
            "Compare performance across model versions and threshold settings.",
            "Track this scenario as an AI perception failure case.",
            "Include this case in future validation, monitoring, and model-change evaluation.",
        ],
        evidence_chain=[
            f"Low-confidence expected objects: {_format_counts(input_data.low_confidence_expected_objects)}",
            f"Missed expected objects: {_format_counts(input_data.missed_objects)}",
            f"Confidence scores: {input_data.confidence_scores}",
            "AI safety concern: expected objects do not have robust detection evidence in this scenario.",
        ],
    )


def _infer_iso_26262(input_data: SafetyLensInput) -> StandardFinding:
    severity = "LOW"
    if any(label in VULNERABLE_ROAD_USERS for label in input_data.missed_objects):
        severity = "HIGH"
    elif input_data.low_confidence_expected_objects:
        severity = "MEDIUM" if not any(
            label in VULNERABLE_ROAD_USERS for label in input_data.low_confidence_expected_objects
        ) else "HIGH"

    return StandardFinding(
        standard="ISO 26262",
        severity=severity,
        interpretation=(
            "This issue may become relevant to functional safety if downstream safety functions depend on perception "
            "outputs for braking, warning, or trajectory planning. The main concern is not the AI limitation itself, "
            "but whether the system-level safety concept can detect, tolerate, or mitigate perception failure."
        ),
        recommendations=[
            "Check whether safety requirements exist for handling missed or low-confidence perception outputs.",
            "Review whether fallback or degradation strategies are defined.",
            "Verify whether perception confidence is monitored by downstream functions.",
            "Assess whether missed vulnerable road users can contribute to hazardous behavior.",
            "Link this failure case to relevant safety goals, technical safety requirements, or verification activities if applicable.",
        ],
        evidence_chain=[
            f"Detected above display threshold: {_format_counts(input_data.detected_objects)}",
            f"Low-confidence expected objects: {_format_counts(input_data.low_confidence_expected_objects)}",
            f"Missed expected objects: {_format_counts(input_data.missed_objects)}",
            "Functional safety concern: downstream safety behavior should not assume missing or weak perception is safe.",
        ],
    )


def evaluate_safety_lens(
    raw_detections: Iterable[Detection],
    display_detections: Iterable[Detection],
    expected_objects: dict[str, int],
    display_threshold: float,
    low_conf_threshold: float,
    scenario_tags: list[str] | None = None,
    metrics: dict | None = None,
) -> SafetyLensV2Result:
    input_data = _build_input(
        raw_detections=list(raw_detections),
        display_detections=list(display_detections),
        expected_objects=expected_objects,
        display_threshold=display_threshold,
        low_conf_threshold=low_conf_threshold,
        scenario_tags=scenario_tags,
    )

    findings = [
        _infer_sotif(input_data),
        _infer_iso_8800(input_data),
        _infer_iso_26262(input_data),
    ]
    severity = _highest_severity(findings)
    primary_safety_concern = _build_primary_concern(input_data)
    observed_issues = _build_observed_issues(input_data, metrics=metrics)

    standard_specific_recommendations: list[str] = []
    for finding in findings:
        standard_specific_recommendations.extend(finding.recommendations)
    standard_specific_recommendations = list(dict.fromkeys(standard_specific_recommendations))

    evidence_chain: list[str] = []
    for finding in findings:
        evidence_chain.extend(finding.evidence_chain)
    evidence_chain = list(dict.fromkeys(evidence_chain))

    return SafetyLensV2Result(
        severity=severity,
        expected_objects=input_data.expected_objects,
        detected_objects=input_data.detected_objects,
        low_confidence_expected_objects=input_data.low_confidence_expected_objects,
        missed_expected_objects=input_data.missed_objects,
        scenario_tags=input_data.scenario_tags,
        primary_safety_concern=primary_safety_concern,
        observed_perception_issues=observed_issues,
        standard_specific_findings=findings,
        standard_specific_recommendations=standard_specific_recommendations,
        evidence_chain=evidence_chain,
    )


def _recommended_follow_up_tests(result: SafetyLensV2Result) -> list[str]:
    tests: list[str] = []
    if "night" in result.scenario_tags:
        tests.append("Nighttime pedestrian crossing with different pedestrian distances.")
    if "occlusion" in result.scenario_tags or any(
        label in VULNERABLE_ROAD_USERS for label in result.low_confidence_expected_objects
    ):
        tests.append("Pedestrian partially occluded by vehicle or street furniture.")
    if any(label in VULNERABLE_ROAD_USERS for label in result.expected_objects):
        tests.append("Multiple pedestrians crossing simultaneously.")
    if "traffic light" in result.expected_objects:
        tests.append("Traffic light detection under glare or low-light conditions.")
    tests.extend(
        [
            "Threshold sensitivity test from 0.25 to 0.90.",
            "Model comparison test between YOLOv8n, YOLO11n, YOLO11s, and larger variants.",
        ]
    )
    return list(dict.fromkeys(tests))


def _format_metric(metric_name: str, metrics: dict | None) -> str:
    if not metrics:
        return "N/A"
    value = metrics.get(metric_name)
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.2f}" if metric_name in {"precision", "recall", "display_threshold", "low_confidence_threshold", "map50", "map50_95"} else str(value)
    return str(value)


def generate_safety_report(
    result: SafetyLensV2Result,
    scenario_name: str = "",
    metrics: dict | None = None,
) -> str:
    findings_by_standard = {finding.standard: finding for finding in result.standard_specific_findings}
    follow_up_tests = _recommended_follow_up_tests(result)

    lines = [
        "## Safety Lens Assessment",
        f"Severity: {result.severity}",
        "",
        "Primary safety concern:",
        result.primary_safety_concern,
        "",
        "Observed perception issues:",
    ]

    for index, issue in enumerate(result.observed_perception_issues, start=1):
        lines.append(f"{index}. {issue}")

    lines.extend(
        [
            "",
            "Standard-specific interpretation:",
            "",
            "ISO 21448 / SOTIF:",
            findings_by_standard["ISO 21448 / SOTIF"].interpretation,
            "",
            "Recommended SOTIF actions:",
        ]
    )
    lines.extend(f"- {item}" for item in findings_by_standard["ISO 21448 / SOTIF"].recommendations)

    lines.extend(
        [
            "",
            "ISO 8800:",
            findings_by_standard["ISO 8800"].interpretation,
            "",
            "Recommended ISO 8800-style actions:",
        ]
    )
    lines.extend(f"- {item}" for item in findings_by_standard["ISO 8800"].recommendations)

    lines.extend(
        [
            "",
            "ISO 26262:",
            findings_by_standard["ISO 26262"].interpretation,
            "",
            "Recommended ISO 26262-style actions:",
        ]
    )
    lines.extend(f"- {item}" for item in findings_by_standard["ISO 26262"].recommendations)

    lines.extend(["", "Recommended follow-up tests:"])
    lines.extend(f"- {item}" for item in follow_up_tests)

    lines.extend(
        [
            "",
            "Evidence summary:",
            f"- Scenario: {scenario_name or 'Unspecified scene'}",
            f"- Scenario tags: {', '.join(result.scenario_tags) if result.scenario_tags else 'None inferred'}",
            f"- Expected objects: {_format_counts(result.expected_objects)}",
            f"- Detected objects above threshold: {_format_counts(result.detected_objects)}",
            f"- Low-confidence expected objects: {_format_counts(result.low_confidence_expected_objects)}",
            f"- Missed expected objects: {_format_counts(result.missed_expected_objects)}",
            f"- Precision: {_format_metric('precision', metrics)}",
            f"- Recall: {_format_metric('recall', metrics)}",
            f"- Detection threshold: {_format_metric('display_threshold', metrics)}",
            f"- Low-confidence threshold: {_format_metric('low_confidence_threshold', metrics)}",
            f"- mAP50: {_format_metric('map50', metrics)}",
            f"- mAP50-95: {_format_metric('map50_95', metrics)}",
            (
                "- Ground-truth boxes: available for IoU / mAP analysis."
                if metrics and metrics.get("ground_truth_boxes_available")
                else "- Ground-truth boxes: not available, so IoU and mAP metrics cannot be used for this single-image report."
            ),
            "",
            "Evidence chain:",
        ]
    )
    lines.extend(f"- {item}" for item in result.evidence_chain)

    conclusion = (
        "The current result should not be treated as a release-blocking conclusion by itself, but it should be treated "
        "as a safety-relevant perception evidence item. It should be added to the failure-case library and used for "
        "threshold sensitivity analysis, scenario coverage expansion, and future regression testing."
        if result.severity in {"CRITICAL", "HIGH", "MEDIUM"}
        else "The current result does not show a major expected-object issue at the selected threshold, but it should still remain part of regression monitoring."
    )
    lines.extend(["", "Final conclusion:", conclusion])

    return "\n".join(lines)

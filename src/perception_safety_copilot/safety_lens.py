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
    evidence_sufficiency: str
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
    assessment_limitations: list[str]
    operating_recommendation: str


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


def _labels(counts: dict[str, int]) -> set[str]:
    return {label for label, count in counts.items() if count > 0}


def _merge_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _combined_issue_labels(input_data: SafetyLensInput) -> set[str]:
    return _labels(input_data.low_confidence_expected_objects) | _labels(input_data.missed_objects)


def _dynamic_sotif_recommendations(input_data: SafetyLensInput) -> list[str]:
    issue_labels = _combined_issue_labels(input_data)
    recs = ["Classify this as a potential triggering condition."]

    if "night" in input_data.scenario_tags:
        recs.append("Add more nighttime cases across different illumination levels and sensor exposure conditions.")
    if "crosswalk" in input_data.scenario_tags:
        recs.append("Create regression tests for pedestrian-crossing and crosswalk-entry scenarios.")
    if "occlusion" in input_data.scenario_tags:
        recs.append("Evaluate performance under partial occlusion, late emergence, and truncated object views.")
    if "glare" in input_data.scenario_tags or "traffic light" in issue_labels:
        recs.append("Stress-test the model under glare, reflective surfaces, and difficult traffic-signal visibility.")

    if any(label in VULNERABLE_ROAD_USERS for label in _labels(input_data.missed_objects)):
        recs.append("Review whether this scenario should trigger conservative behavior or ODD restrictions until coverage improves.")
    elif any(label in VULNERABLE_ROAD_USERS for label in _labels(input_data.low_confidence_expected_objects)):
        recs.append("Compare threshold settings to understand when vulnerable road users drop below the operating threshold.")

    if issue_labels & IMPORTANT_OBJECTS:
        recs.append("Slice validation by object size, distance, and scene geometry for safety-relevant road objects.")

    recs.append("Evaluate performance across illumination, distance, pose, and occlusion variations.")
    return _merge_unique(recs)


def _dynamic_iso_8800_recommendations(input_data: SafetyLensInput) -> list[str]:
    issue_labels = _combined_issue_labels(input_data)
    recs = ["Track this scenario as an AI perception failure case."]

    if any(label in VULNERABLE_ROAD_USERS for label in issue_labels):
        recs.append("Review dataset coverage and class balance for vulnerable road users in this scenario family.")
        recs.append("Analyze confidence calibration specifically for vulnerable road users and near-threshold detections.")

    if "night" in input_data.scenario_tags:
        recs.append("Audit training and validation coverage for nighttime and low-light scenes.")
    if "rain" in input_data.scenario_tags or "fog" in input_data.scenario_tags:
        recs.append("Check robustness under adverse visibility conditions and compare model behavior across weather slices.")
    if "traffic light" in issue_labels:
        recs.append("Review dataset coverage for traffic lights, signal visibility, and long-range small-object detection.")
    if any(label in IMPORTANT_OBJECTS for label in issue_labels):
        recs.append("Compare performance across model versions and threshold settings for safety-relevant object classes.")

    recs.append("Include this case in future validation, monitoring, and model-change evaluation.")
    return _merge_unique(recs)


def _dynamic_iso_26262_recommendations(input_data: SafetyLensInput) -> list[str]:
    issue_labels = _combined_issue_labels(input_data)
    recs = ["Check whether safety requirements exist for handling missed or low-confidence perception outputs."]

    if any(label in VULNERABLE_ROAD_USERS for label in _labels(input_data.missed_objects)):
        recs.append("Assess whether missed vulnerable road users can contribute to hazardous braking, steering, or trajectory decisions.")
        recs.append("Link this failure case to safety goals, technical safety requirements, and verification evidence for pedestrian protection.")
    elif any(label in VULNERABLE_ROAD_USERS for label in _labels(input_data.low_confidence_expected_objects)):
        recs.append("Verify whether downstream functions monitor perception confidence and enter degraded behavior when VRU evidence is weak.")

    if "traffic light" in issue_labels:
        recs.append("Review how downstream logic behaves when traffic-signal perception is missing, stale, or uncertain.")
    if any(label in IMPORTANT_OBJECTS for label in issue_labels):
        recs.append("Review whether fallback or degradation strategies are defined for uncertain perception of safety-relevant road objects.")

    recs.append("Verify whether perception confidence is monitored by downstream functions.")
    return _merge_unique(recs)


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
    if metrics and metrics.get("enhancement_failure_detected"):
        issues.append(
            "Preprocessing did not improve perception performance. This suggests the scenario requires model-level robustness improvement or adverse-weather training data."
        )

    if not issues:
        issues.append("No expected-object miss or major low-confidence issue was found for this scene.")

    return issues


def _scenario_condition_text(input_data: SafetyLensInput) -> str:
    labels = {
        "night": "nighttime illumination",
        "rain": "rain and reduced visibility",
        "fog": "fog or haze",
        "glare": "glare",
        "crosswalk": "crosswalk geometry",
        "occlusion": "partial occlusion",
        "intersection": "intersection complexity",
        "urban": "urban scene complexity",
    }
    conditions = [label for tag, label in labels.items() if tag in input_data.scenario_tags]
    return ", ".join(conditions) if conditions else "the supplied operating scene"


def _failure_description(input_data: SafetyLensInput) -> str:
    parts: list[str] = []
    if input_data.missed_objects:
        parts.append(f"complete misses ({_format_counts(input_data.missed_objects)})")
    if input_data.low_confidence_expected_objects:
        parts.append(
            "threshold-sensitive candidates "
            f"({_format_counts(input_data.low_confidence_expected_objects)})"
        )
    return " and ".join(parts) if parts else "no expected-object failure"


def _infer_sotif(input_data: SafetyLensInput) -> StandardFinding:
    condition_text = _scenario_condition_text(input_data)
    failure_text = _failure_description(input_data)

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
            f"The observed {failure_text} under {condition_text} is consistent with a potential perception "
            "case where perception performance is insufficient. The evidence supports investigating these "
            "conditions as possible SOTIF "
            "triggering conditions; it does not by itself establish the root cause."
        ),
        recommendations=_dynamic_sotif_recommendations(input_data),
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
            f"The AI-performance evidence contains {_failure_description(input_data)} under "
            f"{_scenario_condition_text(input_data)}. This creates concrete questions about scenario coverage, "
            "class balance, confidence calibration, and robustness. Dataset or calibration weakness remains a "
            "hypothesis until slice-level evidence is reviewed."
        ),
        recommendations=_dynamic_iso_8800_recommendations(input_data),
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
            f"The perception evidence shows {_failure_description(input_data)}. ISO 26262 relevance depends on the "
            "vehicle function that consumes this output and whether the failure can contribute to hazardous behavior. "
            "A single image cannot establish Exposure, Controllability, or ASIL; it can identify a failure case that "
            "must be traced to safety goals, monitoring, fallback, and verification evidence."
        ),
        recommendations=_dynamic_iso_26262_recommendations(input_data),
        evidence_chain=[
            f"Detected above display threshold: {_format_counts(input_data.detected_objects)}",
            f"Low-confidence expected objects: {_format_counts(input_data.low_confidence_expected_objects)}",
            f"Missed expected objects: {_format_counts(input_data.missed_objects)}",
            "Functional safety concern: downstream safety behavior should not assume missing or weak perception is safe.",
        ],
    )


def _assessment_limitations(input_data: SafetyLensInput, metrics: dict | None) -> list[str]:
    limitations: list[str] = []
    if not input_data.expected_objects:
        limitations.append(
            "No expected-object counts were supplied, so missed-object severity and single-image recall cannot be established."
        )
    if not input_data.scenario_tags:
        limitations.append(
            "No verified scenario tags were supplied, so condition-specific conclusions are limited."
        )
    if not metrics or not metrics.get("ground_truth_boxes_available"):
        limitations.append(
            "Ground-truth boxes are unavailable, so localization quality and single-image IoU/mAP cannot be verified."
        )
    if metrics and metrics.get("map50") is None:
        limitations.append(
            "mAP is a dataset-level metric and is not available as evidence for this individual image."
        )
    return limitations


def _evidence_sufficiency(input_data: SafetyLensInput, metrics: dict | None) -> str:
    if not input_data.expected_objects:
        return "LIMITED"
    if not metrics or not metrics.get("ground_truth_boxes_available"):
        return "MODERATE"
    return "STRONG"


def _operating_recommendation(
    severity: str,
    input_data: SafetyLensInput,
    evidence_sufficiency: str,
) -> str:
    if evidence_sufficiency == "LIMITED":
        return (
            "Do not interpret the LOW severity as proof of complete perception. Add reviewed expected objects or "
            "ground-truth annotations before using this image as pass/fail evidence."
        )
    if severity == "CRITICAL":
        return (
            "Treat this image as a priority failure case. Block acceptance of this scenario configuration until the "
            "complete miss is reproduced, reviewed, and covered by regression testing."
        )
    if severity == "HIGH":
        return (
            "Treat this image as safety-relevant threshold sensitivity. Compare model and threshold settings, then "
            "verify downstream handling of weak vulnerable-road-user evidence."
        )
    if severity == "MEDIUM":
        return (
            "Investigate the affected classes before accepting robustness for this scenario family and retain the "
            "image as a regression case."
        )
    return (
        "No major expected-object failure was found at the selected threshold. Retain the case for regression "
        "monitoring; this is not a release conclusion by itself."
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
    evidence_sufficiency = _evidence_sufficiency(input_data, metrics)
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
        evidence_sufficiency=evidence_sufficiency,
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
        assessment_limitations=_assessment_limitations(input_data, metrics),
        operating_recommendation=_operating_recommendation(
            severity,
            input_data,
            evidence_sufficiency,
        ),
    )


def _recommended_follow_up_tests(result: SafetyLensV2Result) -> list[str]:
    tests: list[str] = []
    if "night" in result.scenario_tags:
        tests.append("Repeat the scene across measured illumination levels and object distances.")
    if "rain" in result.scenario_tags:
        tests.append("Repeat the scene across rain intensity, windshield contamination, and spray levels.")
    if "fog" in result.scenario_tags:
        tests.append("Evaluate the affected classes across controlled visibility ranges.")
    if "glare" in result.scenario_tags:
        tests.append("Sweep glare source position and intensity while preserving the same scene geometry.")
    if "occlusion" in result.scenario_tags or any(
        label in VULNERABLE_ROAD_USERS for label in result.low_confidence_expected_objects
    ):
        tests.append("Vary occlusion ratio and emergence timing for the affected vulnerable road users.")
    if any(label in VULNERABLE_ROAD_USERS for label in result.expected_objects):
        tests.append("Vary vulnerable-road-user count, pose, scale, and lateral position.")
    if "traffic light" in result.expected_objects:
        tests.append("Vary traffic-light distance, signal state, glare, and partial obstruction.")
    if result.low_confidence_expected_objects:
        affected = ", ".join(sorted(result.low_confidence_expected_objects))
        tests.append(f"Run a threshold sweep for the near-threshold classes: {affected}.")
    if result.missed_expected_objects:
        affected = ", ".join(sorted(result.missed_expected_objects))
        tests.append(f"Compare model variants on the completely missed classes: {affected}.")
    if not tests:
        tests.append("Retain this scene in the regression set and monitor confidence drift across model changes.")
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


def _robustness_conclusion(result: SafetyLensV2Result, metrics: dict | None) -> str:
    if not metrics:
        return "Robustness conclusion is unavailable because no evaluation metrics were provided."

    if metrics.get("enhancement_failure_detected") and metrics.get("best_benchmark_model"):
        return (
            "Classical and learned preprocessing did not materially improve this case. "
            f"The stronger model benchmark suggests `{metrics.get('best_benchmark_model')}` performs best on this disturbance slice, "
            "so the next priority should be model-level robustness improvement and disturbance-rich training data."
        )
    if metrics.get("enhancement_failure_detected"):
        return (
            "Preprocessing did not improve this case, which suggests the current pipeline remains fragile under disturbance. "
            "The next priority should be stronger detection models and adverse-weather or low-light training data."
        )
    if metrics.get("enhancement_best_variant") and metrics.get("enhancement_best_variant") != "Original":
        return (
            f"Preprocessing improved perception evidence for this scene, with `{metrics.get('enhancement_best_variant')}` "
            "performing best. This indicates the disturbance is at least partially recoverable through front-end enhancement."
        )
    return (
        "The current pipeline shows baseline robustness for this scene, but broader disturbance-slice benchmarking is still needed "
        "before drawing strong deployment conclusions."
    )


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
        f"Evidence sufficiency: {result.evidence_sufficiency}",
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
            "Operating recommendation:",
            result.operating_recommendation,
            "",
            "Assessment limitations:",
        ]
    )
    if result.assessment_limitations:
        lines.extend(f"- {item}" for item in result.assessment_limitations)
    else:
        lines.append("- No major evidence limitation was identified for this evaluation.")

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
            "Robustness conclusion:",
            _robustness_conclusion(result, metrics),
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
                f"- Enhancement comparison: no improvement over Original; best variant was "
                f"{metrics.get('enhancement_best_variant', 'N/A')}."
                if metrics and metrics.get("enhancement_failure_detected")
                else (
                    f"- Enhancement comparison: best variant was {metrics.get('enhancement_best_variant', 'N/A')} "
                    f"with {metrics.get('enhancement_best_detections', 'N/A')} detections."
                    if metrics and metrics.get("enhancement_best_variant")
                    else "- Enhancement comparison: not run."
                )
            ),
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

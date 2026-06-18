from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .project1_bridge import (
    DEFAULT_ISO_26262_SCHEME,
    DEFAULT_ISO_8800_SCHEME,
    DEFAULT_NUSCENES_PROFILE,
    DEFAULT_PROJECT1_DIR,
    DEFAULT_SOTIF_SCHEME,
)


@dataclass(frozen=True)
class RetrievedContext:
    title: str
    source_path: Path
    layer: str
    score: int
    matched_terms: list[str]
    excerpt: str


@dataclass(frozen=True)
class RetrievalBundle:
    query_terms: list[str]
    similar_scenarios: list[RetrievedContext]
    safety_context: list[RetrievedContext]
    standards_guidance: list[RetrievedContext]


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", _normalize(text)) if len(token) > 2]


def _extract_excerpt(text: str, matched_terms: list[str], radius: int = 260) -> str:
    normalized = _normalize(text)
    match_index = -1
    chosen_term = ""
    for term in matched_terms:
        match_index = normalized.find(term)
        if match_index >= 0:
            chosen_term = term
            break

    if match_index < 0:
        snippet = text[:radius * 2].strip()
        return snippet + ("..." if len(text) > len(snippet) else "")

    start = max(0, match_index - radius)
    end = min(len(text), match_index + len(chosen_term) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _score_document(text: str, title: str, terms: list[str]) -> tuple[int, list[str]]:
    haystack = f"{_normalize(title)}\n{_normalize(text)}"
    score = 0
    matched_terms: list[str] = []
    for term in terms:
        if term in haystack:
            score += 3
            matched_terms.append(term)
            if term in _normalize(title):
                score += 3
    return score, matched_terms


def _candidate_documents() -> list[tuple[str, str, Path]]:
    standards_dir = DEFAULT_PROJECT1_DIR / "standards_pdfs"
    return [
        ("similar_scenarios", "AEB Pedestrian Safety Case", standards_dir / "project_example_aeb_pedestrian_safety_case.md"),
        ("similar_scenarios", "Lane Maintaining Perception Safety Case", standards_dir / "project_example_lane_maintaining_perception_safety_case.md"),
        ("similar_scenarios", "LiDAR Perception Safety Case", standards_dir / "project_example_lidar_perception_safety_case.md"),
        ("safety_context", "nuScenes Dataset Safety Profile", DEFAULT_NUSCENES_PROFILE),
        ("standards_guidance", "ISO 26262 Evaluation Scheme", DEFAULT_ISO_26262_SCHEME),
        ("standards_guidance", "ISO 21448 / SOTIF Evaluation Scheme", DEFAULT_SOTIF_SCHEME),
        ("standards_guidance", "ISO 8800 Evaluation Scheme", DEFAULT_ISO_8800_SCHEME),
    ]


def build_query_terms(
    scenario_name: str,
    scenario_tags: list[str],
    expected_objects: dict[str, int],
    low_confidence_expected_objects: dict[str, int],
    missed_expected_objects: dict[str, int],
) -> list[str]:
    terms: list[str] = []
    terms.extend(_tokenize(scenario_name))
    terms.extend(_tokenize(" ".join(scenario_tags)))
    terms.extend(_tokenize(" ".join(expected_objects.keys())))
    terms.extend(_tokenize(" ".join(low_confidence_expected_objects.keys())))
    terms.extend(_tokenize(" ".join(missed_expected_objects.keys())))

    if any(label in {"person", "pedestrian", "cyclist", "bicycle", "motorcycle"} for label in missed_expected_objects):
        terms.extend(["vru", "pedestrian", "occlusion"])
    if "night" in scenario_tags:
        terms.extend(["night", "glare"])
    if "crosswalk" in scenario_tags:
        terms.extend(["crossing", "urban"])
    if "traffic_light" in scenario_tags or "traffic light" in expected_objects:
        terms.extend(["traffic", "signal"])

    return list(dict.fromkeys(term for term in terms if term))


def retrieve_project1_evidence(
    scenario_name: str,
    scenario_tags: list[str],
    expected_objects: dict[str, int],
    low_confidence_expected_objects: dict[str, int],
    missed_expected_objects: dict[str, int],
    top_k_similar: int = 3,
) -> RetrievalBundle:
    query_terms = build_query_terms(
        scenario_name,
        scenario_tags,
        expected_objects,
        low_confidence_expected_objects,
        missed_expected_objects,
    )

    matches: list[RetrievedContext] = []
    for layer, title, path in _candidate_documents():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        score, matched_terms = _score_document(text, title, query_terms)
        if score == 0 and layer != "standards_guidance":
            continue
        matches.append(
            RetrievedContext(
                title=title,
                source_path=path,
                layer=layer,
                score=score,
                matched_terms=matched_terms,
                excerpt=_extract_excerpt(text, matched_terms),
            )
        )

    similar_scenarios = sorted(
        [match for match in matches if match.layer == "similar_scenarios"],
        key=lambda match: match.score,
        reverse=True,
    )[:top_k_similar]
    safety_context = sorted(
        [match for match in matches if match.layer == "safety_context"],
        key=lambda match: match.score,
        reverse=True,
    )[:1]
    standards_guidance = sorted(
        [match for match in matches if match.layer == "standards_guidance"],
        key=lambda match: match.score,
        reverse=True,
    )[:3]

    return RetrievalBundle(
        query_terms=query_terms,
        similar_scenarios=similar_scenarios,
        safety_context=safety_context,
        standards_guidance=standards_guidance,
    )


def render_retrieval_markdown(bundle: RetrievalBundle) -> str:
    lines = [
        "### Scenario Retrieval Layer",
        "",
        f"- Query terms: {', '.join(bundle.query_terms) if bundle.query_terms else 'None'}",
        "",
        "#### Similar Known Scenarios",
    ]

    if not bundle.similar_scenarios:
        lines.append("- No similar Project 1 scenario document was retrieved.")
    else:
        for item in bundle.similar_scenarios:
            lines.extend(
                [
                    f"- **{item.title}** (`score={item.score}`)",
                    f"  - Matched terms: {', '.join(item.matched_terms) if item.matched_terms else 'None'}",
                    f"  - Source: `{item.source_path.name}`",
                    f"  - Excerpt: {item.excerpt}",
                ]
            )

    lines.extend(["", "#### Relevant Safety Context"])
    if not bundle.safety_context:
        lines.append("- No Project 1 safety-context document was retrieved.")
    else:
        for item in bundle.safety_context:
            lines.extend(
                [
                    f"- **{item.title}** (`score={item.score}`)",
                    f"  - Source: `{item.source_path.name}`",
                    f"  - Excerpt: {item.excerpt}",
                ]
            )

    lines.extend(["", "#### Project 1 Standards Guidance"])
    if not bundle.standards_guidance:
        lines.append("- No standards guidance document was retrieved.")
    else:
        for item in bundle.standards_guidance:
            lines.extend(
                [
                    f"- **{item.title}** (`score={item.score}`)",
                    f"  - Source: `{item.source_path.name}`",
                    f"  - Excerpt: {item.excerpt}",
                ]
            )

    return "\n".join(lines)

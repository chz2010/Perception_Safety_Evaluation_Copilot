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
    evidence_id: str
    title: str
    source_path: Path
    layer: str
    score: int
    matched_terms: list[str]
    excerpt: str
    retrieval_reason: str


@dataclass(frozen=True)
class RetrievalBundle:
    query_terms: list[str]
    query_plan: dict[str, list[str]]
    grounding_notes: list[str]
    similar_scenarios: list[RetrievedContext]
    failure_mechanisms: list[RetrievedContext]
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


def _passages(text: str) -> list[str]:
    passages: list[str] = []
    current_heading = ""
    for block in re.split(r"\n\s*\n", text):
        cleaned = block.strip()
        if not cleaned:
            continue
        if cleaned.startswith("#"):
            current_heading = cleaned
            continue
        passage = f"{current_heading}\n{cleaned}".strip()
        if len(passage) >= 60:
            passages.append(passage)
    return passages or [text]


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


def _best_passage(text: str, title: str, terms: list[str]) -> tuple[int, list[str], str]:
    best_score = 0
    best_terms: list[str] = []
    best_text = text
    for passage in _passages(text):
        score, matched_terms = _score_document(passage, title, terms)
        if score > best_score:
            best_score = score
            best_terms = matched_terms
            best_text = passage
    return best_score, best_terms, _extract_excerpt(best_text, best_terms)


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


def build_query_plan(
    scenario_name: str,
    scenario_tags: list[str],
    detected_objects: dict[str, int],
    expected_objects: dict[str, int],
    low_confidence_expected_objects: dict[str, int],
    missed_expected_objects: dict[str, int],
) -> dict[str, list[str]]:
    scenario_terms = _tokenize(scenario_name) + _tokenize(" ".join(scenario_tags))
    failure_terms = (
        _tokenize(" ".join(low_confidence_expected_objects.keys()))
        + _tokenize(" ".join(missed_expected_objects.keys()))
    )
    observed_object_terms = _tokenize(" ".join(detected_objects.keys()))
    expected_object_terms = _tokenize(" ".join(expected_objects.keys()))

    if missed_expected_objects:
        failure_terms.extend(["missed", "detection", "false negative", "insufficient performance"])
    if low_confidence_expected_objects:
        failure_terms.extend(["confidence", "uncertainty", "threshold", "calibration"])
    if any(
        label in {"person", "pedestrian", "cyclist", "bicycle", "motorcycle"}
        for label in set(missed_expected_objects) | set(low_confidence_expected_objects)
    ):
        failure_terms.extend(["vru", "pedestrian", "vulnerable road user", "occlusion"])
    if "night" in scenario_tags:
        scenario_terms.extend(["night", "low light", "illumination"])
    if "crosswalk" in scenario_tags:
        scenario_terms.extend(["crossing", "urban", "pedestrian"])
    if "rain" in scenario_tags or "fog" in scenario_tags:
        scenario_terms.extend(["weather", "visibility", "environmental condition"])
    if "traffic_light" in scenario_tags or "traffic light" in expected_objects:
        scenario_terms.extend(["traffic", "signal"])

    shared_failure_terms = list(
        dict.fromkeys(failure_terms + scenario_terms + observed_object_terms + expected_object_terms)
    )
    return {
        "scenario_similarity": list(dict.fromkeys(term for term in scenario_terms if term)),
        "failure_mechanism": list(dict.fromkeys(term for term in shared_failure_terms if term)),
        "sotif": list(
            dict.fromkeys(
                shared_failure_terms
                + ["triggering condition", "functional insufficiency", "scenario coverage", "sotif"]
            )
        ),
        "iso_8800": list(
            dict.fromkeys(
                shared_failure_terms
                + ["data quality", "dataset coverage", "robustness", "model performance", "ai safety"]
            )
        ),
        "iso_26262": list(
            dict.fromkeys(
                shared_failure_terms
                + ["hazard", "safety goal", "fallback", "degradation", "controllability", "asil"]
            )
        ),
    }


def build_query_terms(
    scenario_name: str,
    scenario_tags: list[str],
    detected_objects: dict[str, int],
    expected_objects: dict[str, int],
    low_confidence_expected_objects: dict[str, int],
    missed_expected_objects: dict[str, int],
) -> list[str]:
    plan = build_query_plan(
        scenario_name,
        scenario_tags,
        detected_objects,
        expected_objects,
        low_confidence_expected_objects,
        missed_expected_objects,
    )
    return list(dict.fromkeys(term for terms in plan.values() for term in terms))


def _retrieve(
    documents: list[tuple[str, str, Path]],
    terms: list[str],
    allowed_layers: set[str],
    evidence_prefix: str,
    reason: str,
    top_k: int,
    include_zero_score: bool = False,
    min_score: int = 1,
    min_matched_terms: int = 1,
) -> list[RetrievedContext]:
    matches: list[RetrievedContext] = []
    for layer, title, path in documents:
        if layer not in allowed_layers or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        score, matched_terms, excerpt = _best_passage(text, title, terms)
        if not include_zero_score and (
            score < min_score or len(set(matched_terms)) < min_matched_terms
        ):
            continue
        matches.append(
            RetrievedContext(
                evidence_id="",
                title=title,
                source_path=path,
                layer=layer,
                score=score,
                matched_terms=matched_terms,
                excerpt=excerpt,
                retrieval_reason=reason,
            )
        )

    ranked = sorted(matches, key=lambda match: match.score, reverse=True)[:top_k]
    return [
        RetrievedContext(
            evidence_id=f"{evidence_prefix}-{index}",
            title=item.title,
            source_path=item.source_path,
            layer=item.layer,
            score=item.score,
            matched_terms=item.matched_terms,
            excerpt=item.excerpt,
            retrieval_reason=item.retrieval_reason,
        )
        for index, item in enumerate(ranked, start=1)
    ]


def retrieve_project1_evidence(
    scenario_name: str,
    scenario_tags: list[str],
    expected_objects: dict[str, int],
    low_confidence_expected_objects: dict[str, int],
    missed_expected_objects: dict[str, int],
    detected_objects: dict[str, int] | None = None,
    top_k_similar: int = 3,
) -> RetrievalBundle:
    query_plan = build_query_plan(
        scenario_name,
        scenario_tags,
        detected_objects or {},
        expected_objects,
        low_confidence_expected_objects,
        missed_expected_objects,
    )
    documents = _candidate_documents()
    camera_documents = [item for item in documents if "LiDAR" not in item[1]]
    scenario_term_set = set(query_plan["scenario_similarity"] + query_plan["failure_mechanism"])
    relevant_scenario_documents: list[tuple[str, str, Path]] = []
    if scenario_term_set & {
        "person",
        "pedestrian",
        "cyclist",
        "bicycle",
        "motorcycle",
        "vru",
        "crosswalk",
        "crossing",
    }:
        relevant_scenario_documents.extend(
            item for item in camera_documents if item[1] == "AEB Pedestrian Safety Case"
        )
    if scenario_term_set & {"lane", "marking", "steering", "road edge", "lane keeping"}:
        relevant_scenario_documents.extend(
            item for item in camera_documents if item[1] == "Lane Maintaining Perception Safety Case"
        )
    relevant_scenario_documents = list(dict.fromkeys(relevant_scenario_documents))
    relevant_failure_documents = relevant_scenario_documents + [
        item for item in camera_documents if item[0] == "safety_context"
    ]

    grounding_notes: list[str] = []
    if query_plan["scenario_similarity"] and relevant_scenario_documents:
        similar_scenarios = _retrieve(
            relevant_scenario_documents,
            query_plan["scenario_similarity"],
            {"similar_scenarios"},
            "SCN",
            "Matched explicit scene description and operating-condition terms.",
            top_k_similar,
            min_score=12,
            min_matched_terms=2,
        )
        if not similar_scenarios:
            grounding_notes.append(
                "No Project 1 scenario met the minimum relevance threshold; no scenario analogy was forced."
            )
    elif not query_plan["scenario_similarity"]:
        similar_scenarios = []
        grounding_notes.append(
            "No explicit scene description or inferred scenario tag was available, so scenario retrieval was skipped."
        )
    else:
        similar_scenarios = []
        grounding_notes.append(
            "Project 1 has no scenario example for the grounded scene type, so no analogy was forced."
        )

    has_failure_evidence = bool(low_confidence_expected_objects or missed_expected_objects)
    failure_mechanisms = _retrieve(
        relevant_failure_documents,
        query_plan["failure_mechanism"],
        {"similar_scenarios", "safety_context"},
        "FAIL",
        "Matched the observed missed-object or low-confidence failure pattern.",
        3,
        min_score=12,
        min_matched_terms=2,
    ) if has_failure_evidence else []
    safety_context = _retrieve(
        documents,
        query_plan["failure_mechanism"],
        {"safety_context"},
        "CTX",
        "Matched dataset limitations, coverage, or perception-evaluation context.",
        1,
        min_score=6,
        min_matched_terms=2,
    ) if has_failure_evidence else []
    if not has_failure_evidence:
        grounding_notes.append(
            "No expected-object miss or low-confidence expected object was present, so failure retrieval was skipped."
        )

    standard_specs = [
        ("ISO 21448 / SOTIF", "sotif", "STD-SOTIF"),
        ("ISO 8800", "iso_8800", "STD-8800"),
        ("ISO 26262", "iso_26262", "STD-26262"),
    ]
    standards_guidance: list[RetrievedContext] = []
    for title_fragment, query_key, evidence_prefix in standard_specs:
        if not has_failure_evidence:
            continue
        standard_documents = [
            item for item in documents if item[0] == "standards_guidance" and title_fragment in item[1]
        ]
        standards_guidance.extend(
            _retrieve(
                standard_documents,
                query_plan[query_key],
                {"standards_guidance"},
                evidence_prefix,
                f"Retrieved specifically for {title_fragment} interpretation of this failure.",
                1,
                include_zero_score=True,
            )
        )

    query_terms = list(dict.fromkeys(term for terms in query_plan.values() for term in terms))

    return RetrievalBundle(
        query_terms=query_terms,
        query_plan=query_plan,
        grounding_notes=grounding_notes,
        similar_scenarios=similar_scenarios,
        failure_mechanisms=failure_mechanisms,
        safety_context=safety_context,
        standards_guidance=standards_guidance,
    )


def render_retrieval_markdown(bundle: RetrievalBundle) -> str:
    lines = [
        "### Supporting Safety Evidence",
        "",
        "Independent retrieval is used so scene similarity, failure mechanisms, and standards guidance do not compete in one search.",
    ]
    if bundle.grounding_notes:
        lines.extend(["", "#### Grounding Status"])
        lines.extend(f"- {note}" for note in bundle.grounding_notes)
    lines.extend(
        [
        "",
        "#### Similar Known Scenarios",
        ]
    )

    if not bundle.similar_scenarios:
        lines.append("- No similar Project 1 scenario document was retrieved.")
    else:
        for item in bundle.similar_scenarios:
            lines.extend(
                [
                    f"- **[{item.evidence_id}] {item.title}** (`score={item.score}`)",
                    f"  - Why retrieved: {item.retrieval_reason}",
                    f"  - Matched terms: {', '.join(item.matched_terms) if item.matched_terms else 'None'}",
                    f"  - Source: `{item.source_path.name}`",
                    f"  - Excerpt: {item.excerpt}",
                ]
            )

    lines.extend(["", "#### Failure Mechanism Evidence"])
    if not bundle.failure_mechanisms:
        lines.append("- No Project 1 passage matched the observed failure mechanism.")
    else:
        for item in bundle.failure_mechanisms:
            lines.extend(
                [
                    f"- **[{item.evidence_id}] {item.title}** (`score={item.score}`)",
                    f"  - Why retrieved: {item.retrieval_reason}",
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
                    f"- **[{item.evidence_id}] {item.title}** (`score={item.score}`)",
                    f"  - Why retrieved: {item.retrieval_reason}",
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
                    f"- **[{item.evidence_id}] {item.title}** (`score={item.score}`)",
                    f"  - Why retrieved: {item.retrieval_reason}",
                    f"  - Source: `{item.source_path.name}`",
                    f"  - Excerpt: {item.excerpt}",
                ]
            )

    return "\n".join(lines)

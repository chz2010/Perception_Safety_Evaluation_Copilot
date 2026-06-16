from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT1_DIR = PROJECT_ROOT / "Autonomous_Driving_Safety_Analyst"
DEFAULT_NUSCENES_PROFILE = DEFAULT_PROJECT1_DIR / "standards_pdfs" / "nuscenes_dataset_profile.md"
DEFAULT_ISO_26262_SCHEME = DEFAULT_PROJECT1_DIR / "standards_pdfs" / "iso_26262_evaluation_scheme.md"
DEFAULT_SOTIF_SCHEME = DEFAULT_PROJECT1_DIR / "standards_pdfs" / "sotif_evaluation_scheme.md"
DEFAULT_ISO_8800_SCHEME = DEFAULT_PROJECT1_DIR / "standards_pdfs" / "iso_8800_evaluation_scheme.md"


def load_text_excerpt(path: Path, max_chars: int = 1200) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_nuscenes_safety_profile(profile_path: Path = DEFAULT_NUSCENES_PROFILE, max_chars: int = 3500) -> str:
    return load_text_excerpt(profile_path, max_chars=max_chars)


def load_project1_standard_context(max_chars_per_doc: int = 900) -> dict[str, str]:
    return {
        "ISO 26262": load_text_excerpt(DEFAULT_ISO_26262_SCHEME, max_chars=max_chars_per_doc),
        "ISO 21448 / SOTIF": load_text_excerpt(DEFAULT_SOTIF_SCHEME, max_chars=max_chars_per_doc),
        "ISO 8800": load_text_excerpt(DEFAULT_ISO_8800_SCHEME, max_chars=max_chars_per_doc),
    }


def build_project1_context_section(profile_text: str, query: str) -> str:
    if not profile_text:
        return (
            "## Project 1 Safety Context\n\n"
            "- Project 1 nuScenes profile was not found locally. Start Project 1 MCP or regenerate "
            "`standards_pdfs/nuscenes_dataset_profile.md` to enrich this section.\n"
        )

    return "\n".join(
        [
            "## Project 1 Safety Context",
            "",
            f"- Context query: {query}",
            "- Source: Project 1 nuScenes Dataset Safety Profile.",
            "",
            "```text",
            profile_text,
            "```",
        ]
    )


def build_project1_standards_section(standard_context: dict[str, str]) -> str:
    available = {name: text for name, text in standard_context.items() if text}
    if not available:
        return (
            "## Project 1 Standards Context\n\n"
            "- Project 1 standards summaries were not found locally. Rebuild or ingest the standards scheme files to enrich this section.\n"
        )

    lines = ["## Project 1 Standards Context", ""]
    for name, text in available.items():
        lines.extend(
            [
                f"### {name}",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip()

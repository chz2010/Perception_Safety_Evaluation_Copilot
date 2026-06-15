from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT1_DIR = PROJECT_ROOT / "Autonomous_Driving_Safety_Analyst"
DEFAULT_NUSCENES_PROFILE = DEFAULT_PROJECT1_DIR / "standards_pdfs" / "nuscenes_dataset_profile.md"


def load_nuscenes_safety_profile(profile_path: Path = DEFAULT_NUSCENES_PROFILE, max_chars: int = 3500) -> str:
    if not profile_path.exists():
        return ""
    text = profile_path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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


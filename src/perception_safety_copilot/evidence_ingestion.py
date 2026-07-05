"""Index saved Project 3 evaluations as local semantic evidence."""

from __future__ import annotations

from pathlib import Path

from .local_embeddings import LocalEvaluationIndex, build_evaluation_evidence
from .storage import DEFAULT_DB_PATH, load_evaluation, load_evaluations_for_indexing


def index_saved_evaluation(
    evaluation_id: int,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    index: LocalEvaluationIndex | None = None,
    review_notes: str = "",
) -> None:
    record = load_evaluation(evaluation_id, db_path=db_path)
    if record is None:
        raise ValueError(f"Evaluation not found: {evaluation_id}")
    evidence = build_evaluation_evidence(
        evaluation_id=record["id"],
        scenario_name=record["scenario_name"] or "",
        image_name=record["image_name"],
        model_name=record["model_name"],
        metrics=record["metrics"],
        report_markdown=record["report_markdown"],
        review_notes=review_notes,
    )
    (index or LocalEvaluationIndex()).upsert(evidence)


def rebuild_evaluation_index(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    index: LocalEvaluationIndex | None = None,
) -> int:
    target = index or LocalEvaluationIndex()
    records = load_evaluations_for_indexing(db_path=db_path)
    for record in records:
        evidence = build_evaluation_evidence(
            evaluation_id=record["id"],
            scenario_name=record["scenario_name"] or "",
            image_name=record["image_name"],
            model_name=record["model_name"],
            metrics=record["metrics"],
            report_markdown=record["report_markdown"],
        )
        target.upsert(evidence)
    return len(records)

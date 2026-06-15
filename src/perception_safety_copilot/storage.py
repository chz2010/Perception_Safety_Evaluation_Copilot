from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("data/perception_evaluations.sqlite3")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scenario_name TEXT,
                image_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                confidence_threshold REAL NOT NULL,
                low_confidence_threshold REAL NOT NULL,
                detections_json TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                report_markdown TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_evaluation(
    *,
    db_path: Path,
    scenario_name: str,
    image_name: str,
    model_name: str,
    confidence_threshold: float,
    low_confidence_threshold: float,
    detections: list[dict[str, Any]],
    expected: dict[str, int],
    metrics: dict[str, Any],
    report_markdown: str,
) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO evaluations (
                scenario_name,
                image_name,
                model_name,
                confidence_threshold,
                low_confidence_threshold,
                detections_json,
                expected_json,
                metrics_json,
                report_markdown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_name,
                image_name,
                model_name,
                confidence_threshold,
                low_confidence_threshold,
                json.dumps(detections),
                json.dumps(expected),
                json.dumps(metrics),
                report_markdown,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_recent_evaluations(db_path: Path = DEFAULT_DB_PATH, limit: int = 20) -> list[dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, scenario_name, image_name, model_name, metrics_json
            FROM evaluations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["metrics"] = json.loads(record.pop("metrics_json"))
        records.append(record)
    return records


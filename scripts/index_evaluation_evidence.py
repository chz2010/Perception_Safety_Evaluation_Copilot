from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception_safety_copilot.evidence_ingestion import rebuild_evaluation_index
from src.perception_safety_copilot.storage import DEFAULT_DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the local semantic index from saved Project 3 evaluations."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    count = rebuild_evaluation_index(db_path=args.db)
    print(f"Indexed {count} saved evaluations from {args.db}.")


if __name__ == "__main__":
    main()

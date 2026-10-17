"""Detect and refine rescue-object boxes on generated before/after pairs."""

import argparse
from pathlib import Path

from src.core.config import OUTPUT_DIR
from src.evaluation.bounding_box_pipeline import (
    BoundingBoxRunConfig,
    run_bounding_box_pipeline,
)


def parse_args() -> argparse.Namespace:
    output_root = Path(OUTPUT_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--metadata-dir", type=Path, default=output_root / "_metadata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "bounding_boxes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_bounding_box_pipeline(
        BoundingBoxRunConfig(
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    )
    print(
        "Bounding-box run complete: "
        f"{summary['processed_pairs']} pair(s), "
        f"{summary['passed_pairs']} passed, "
        f"{summary['confirmed_rescue_objects']} confirmed object(s).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

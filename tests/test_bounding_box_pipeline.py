import tempfile
import unittest
from pathlib import Path

from src.evaluation.bounding_box_pipeline import (
    BoundingBoxRunConfig,
    load_generation_records,
    run_bounding_box_pipeline,
)
from src.evaluation.yolo_export import (
    DetectionConversionContext,
    _processed_result_to_objects,
)


class BoxValue:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class BoundingBoxPipelineTests(unittest.TestCase):
    def test_load_generation_records_skips_incomplete_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_before = root / "before.png"
            valid_after = root / "after.png"
            valid_before.touch()
            valid_after.touch()

            (root / "valid.json").write_text(
                '{"stem":"valid","reference_path":"%s","output_path":"%s"}'
                % (valid_before.as_posix(), valid_after.as_posix()),
                encoding="utf-8",
            )
            (root / "missing.json").write_text(
                '{"stem":"missing","reference_path":"missing.png",'
                '"output_path":"also-missing.png"}',
                encoding="utf-8",
            )

            records = load_generation_records(root, limit=5)

            self.assertEqual([record["stem"] for record in records], ["valid"])

    def test_load_generation_records_rejects_non_positive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            load_generation_records(Path("unused"), limit=0)

    def test_pipeline_rejects_fewer_complete_pairs_than_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "Only 0 complete"):
                run_bounding_box_pipeline(
                    BoundingBoxRunConfig(
                        metadata_dir=metadata_dir,
                        output_dir=root / "boxes",
                        limit=1,
                    )
                )


class BoundingBoxSlurmTests(unittest.TestCase):
    def test_job_requests_gpu_and_runs_dedicated_entrypoint(self) -> None:
        script = Path("scripts/detect_bounding_boxes.slurm").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --account=uit", script)
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --qos=mig2-35", script)
        self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", script)
        self.assertIn('LIMIT_PAIRS="${1:-5}"', script)
        self.assertIn('"$PYTHON" main_bounding_boxes.py', script)
        self.assertIn("test_bounding_box_pipeline.py", script)
        self.assertIn("test_change_detection.py", script)
        self.assertNotIn("cat .env", script)

    def test_detector_does_not_import_heavy_sdqm_embedding_dependencies(self) -> None:
        source = Path("src/evaluation/yolo_export.py").read_text(encoding="utf-8")

        self.assertNotIn("src.evaluation.sdqm_embedding", source)
        self.assertIn("src.evaluation.image_files", source)


class GroundingDinoConversionTests(unittest.TestCase):
    def test_processed_detection_keeps_class_confidence_and_clips_box(self) -> None:
        objects = _processed_result_to_objects(
            {
                "boxes": [BoxValue([-4.2, 10.1, 105.8, 91.2])],
                "scores": [0.91],
                "text_labels": ["rescue boat"],
            },
            DetectionConversionContext(
                class_map={"rescue_boat": 1},
                image_width=100,
                image_height=100,
                box_threshold=0.25,
            ),
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].class_id, 1)
        self.assertEqual(objects[0].label, "rescue_boat")
        self.assertAlmostEqual(objects[0].confidence, 0.91)
        self.assertEqual(
            (objects[0].box.x_min, objects[0].box.y_min),
            (0, 10),
        )
        self.assertEqual(
            (objects[0].box.x_max, objects[0].box.y_max),
            (100, 92),
        )


if __name__ == "__main__":
    unittest.main()

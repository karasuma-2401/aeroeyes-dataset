import ast
import contextlib
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class GenerationCacheTests(unittest.TestCase):
    def setUp(self):
        source_path = Path(__file__).resolve().parents[1] / "main.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        function_names = {
            "read_json_object",
            "completed_sample",
            "prepare_samples",
        }
        functions = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        ]

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.output = self.root / "output"
        self.downloads = self.root / "downloads"
        (self.output / "_prepared").mkdir(parents=True)
        self.downloads.mkdir()

        class FakeImage:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def convert(self, mode):
                return self

        self.fake_image = FakeImage()
        self.environment = {
            "Path": Path,
            "hashlib": hashlib,
            "json": __import__("json"),
            "torch": SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError)),
            "Image": SimpleNamespace(open=Mock(return_value=self.fake_image)),
            "OUTPUT_DIR": str(self.output),
            "REAL_IMAGES_DIR": str(self.output / "real_reference"),
            "GEN_IMAGES_DIR": str(self.output / "gen_reference"),
            "DOWNLOAD_IMAGES_DIR": self.downloads,
            "LIMIT_IMAGES": 1,
            "IMAGE_SIZE": 512,
            "BASE_SEED": 50,
            "MAX_ATTEMPTS": 4,
            "MAX_CONSECUTIVE_ERRORS": 2,
            "WATERMARK_REMOVAL_ENABLED": False,
            "STAGE_TIMEOUT_SECONDS": 30,
            "budget_exhausted": Mock(return_value=False),
            "error_limit_reached": lambda errors: errors >= 2,
            "load_gemma_model": Mock(return_value=(object(), object())),
            "resize_center_crop": Mock(return_value=self.fake_image),
            "generate_scene_description": Mock(return_value="flooded street"),
            "generate_rescue_instruction": Mock(return_value="add firefighters"),
            "build_flux_prompt": Mock(return_value="flooded street with firefighters"),
            "save_image": Mock(),
            "write_json": Mock(),
            "unload_watermark_tools": Mock(),
            "cleanup_cuda": Mock(),
            "stage": lambda *args: contextlib.nullcontext(),
            "remove_watermark": Mock(),
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"),
            self.environment,
        )

    def test_corrupt_prepared_json_is_rebuilt(self):
        image_key = "flood/example.jpg"
        stem = hashlib.sha256(image_key.encode("utf-8")).hexdigest()
        prepared_path = self.output / "_prepared" / f"{stem}.json"
        reference_path = self.output / "_prepared" / f"{stem}.png"
        prepared_path.write_text("", encoding="utf-8")
        reference_path.touch()

        records = self.environment["prepare_samples"]({
            image_key: {
                "incidents": {"flood": 1},
                "downloaded_file": "example.png",
                "url": "https://example.test/image",
            },
        }, started=0)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stem"], stem)
        self.environment["generate_scene_description"].assert_called_once()
        self.environment["write_json"].assert_called_once_with(prepared_path, records[0])

    def test_corrupt_completed_metadata_is_treated_as_incomplete(self):
        metadata_directory = self.output / "_metadata"
        metadata_directory.mkdir()
        (metadata_directory / "sample.json").write_text("", encoding="utf-8")

        self.assertIsNone(self.environment["completed_sample"]("sample"))


if __name__ == "__main__":
    unittest.main()

"""Evaluation APIs loaded on demand so lightweight stages avoid optional metrics."""

from importlib import import_module


_EXPORTS = {
    "QualityEvaluators": ("quality", "QualityEvaluators"),
    "compute_o_score": ("quality", "compute_o_score"),
    "compute_ssim": ("quality", "compute_ssim"),
    "evaluate_quality": ("quality", "evaluate_quality"),
    "load_evaluators": ("quality", "load_evaluators"),
    "passes_delta_gate": ("quality", "passes_delta_gate"),
    "passes_quality_gate": ("quality", "passes_quality_gate"),
    "BoundingBox": ("change_detection", "BoundingBox"),
    "ChangeDeltaResult": ("change_detection", "ChangeDeltaResult"),
    "compute_otsu_threshold": ("change_detection", "compute_otsu_threshold"),
    "detect_change_delta": ("change_detection", "detect_change_delta"),
    "DetectedObject": ("object_matching", "DetectedObject"),
    "ObjectMatchingReport": ("object_matching", "ObjectMatchingReport"),
    "match_and_refine_objects": ("object_matching", "match_and_refine_objects"),
    "compute_dataset_cmmd": ("cmmd", "compute_dataset_cmmd"),
    "compute_dataset_sdqm": ("sdqm", "compute_dataset_sdqm"),
    "write_sdqm_status_report": ("sdqm", "write_sdqm_status_report"),
    "attach_sdqm_metadata": ("sdqm_metadata", "attach_sdqm_metadata"),
    "write_metadata_jsonl": ("sdqm_metadata", "write_metadata_jsonl"),
    "run_sdqm_regression": ("sdqm_regression", "run_sdqm_regression"),
    "check_custom_ultralytics": ("sdqm_vinfo", "check_custom_ultralytics"),
    "compute_vinfo_metrics": ("sdqm_vinfo", "compute_vinfo_metrics"),
    "export_yolo_dataset": ("yolo_export", "export_yolo_dataset"),
    "export_yolo_pair": ("yolo_export", "export_yolo_pair"),
    "run_dataset_evaluation": ("reporting", "run_dataset_evaluation"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "BoundingBox",
    "ChangeDeltaResult",
    "DetectedObject",
    "ObjectMatchingReport",
    "QualityEvaluators",
    "attach_sdqm_metadata",
    "check_custom_ultralytics",
    "compute_dataset_cmmd",
    "compute_dataset_sdqm",
    "compute_o_score",
    "compute_otsu_threshold",
    "compute_ssim",
    "compute_vinfo_metrics",
    "detect_change_delta",
    "evaluate_quality",
    "export_yolo_dataset",
    "export_yolo_pair",
    "load_evaluators",
    "match_and_refine_objects",
    "passes_delta_gate",
    "passes_quality_gate",
    "run_sdqm_regression",
    "run_dataset_evaluation",
    "write_sdqm_status_report",
    "write_metadata_jsonl",
]

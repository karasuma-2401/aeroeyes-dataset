from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".webp"}


def list_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

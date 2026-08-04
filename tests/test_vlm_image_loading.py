import importlib.util
from pathlib import Path

from PIL import Image


def _load_vlm_utils_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.vlm_utils", root / "openrlhf" / "utils" / "vlm_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_images = _load_vlm_utils_module().load_images


def test_load_images_closes_path_handle_before_returning(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), color="red").save(path)

    images = load_images(str(path))

    assert len(images) == 1
    assert getattr(images[0], "fp", None) is None
    assert images[0].getpixel((0, 0)) == (255, 0, 0)


def test_load_images_preserves_caller_owned_pil_image():
    source = Image.new("RGB", (1, 1), color="blue")

    images = load_images(source)

    assert images == [source]

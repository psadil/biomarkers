from pathlib import Path


def _img_basename(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")

from pathlib import Path
from importlib import resources

import prefect


@prefect.task
def _img_basename(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def get_mpfc_mask() -> Path:
    """Return mPFC mask produced from Smallwood et al. replication.

    Returns:
        Path: Path to mPFC mask.
    """
    with resources.path("biomarkers.data", "smallwood_mpfc_MNI152_1p5.nii.gz") as f:
        mpfc = f
    return mpfc

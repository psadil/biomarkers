from pathlib import Path
import tempfile
import subprocess
import shutil

import prefect

from ..models.fslanat import FSLAnatResult
from .. import utils


def _predict_fsl_anat_output(out: Path, basename: str) -> Path:
    return Path(out / basename).with_suffix(".anat").absolute()


@prefect.task
def _fslanat(image: Path, out: Path):
    basename = utils.img_stem(image)
    anat = _predict_fsl_anat_output(out / "fslanat", basename)

    # if the output already exists, we don't want this to run again.
    # fsl_anat automatically and always adds .anat to the value of -o, so we check for
    # the existence of that predicted output, but then feed in the unmodified value of
    # -o to the task
    if not anat.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfsl = Path(tmpdir) / basename
            subprocess.run(
                ["fsl_anat", "-i", f"{image}", "-o", f"{tmpfsl}"], capture_output=True
            )
            tmpout = _predict_fsl_anat_output(Path(tmpdir), basename)
            FSLAnatResult.from_root(tmpout)
            shutil.copytree(tmpout, anat)


@prefect.flow
def fslanat_flow(images: frozenset[Path], out: Path) -> None:
    _fslanat.map(images, out=out)  # type: ignore

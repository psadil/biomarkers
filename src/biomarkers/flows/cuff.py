from pathlib import Path
import re

import numpy as np
import pandas as pd
import pydantic
from pydantic.dataclasses import dataclass

import nibabel as nb
import ants
from nilearn import image


import prefect
from prefect.tasks import task_input_hash


from .. import utils
from ..task import compcor
from ..task import utils as task_utils


@dataclass(frozen=True)
class CuffFiles:
    bold: pydantic.FilePath
    probseg: frozenset[pydantic.FilePath]
    boldref: pydantic.FilePath
    confounds: pydantic.FilePath
    mask: pydantic.FilePath
    stem: str


@prefect.task
@utils.cache_dataframe
def apply_weights(
    img: Path,
    confounds_file: Path,
    weights: Path,
    brain_mask: Path | None = None,
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:

    confounds = pd.read_parquet(confounds_file)
    n_tr = confounds.shape[0]
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, -n_tr:]
    nii_clean: nb.Nifti1Image = image.clean_img(
        imgs=nii,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        confounds=confounds,
        standardize=False,
        detrend=detrend,
        mask_img=brain_mask,
    )  # type: ignore
    del nii

    weights_raw_nii = nb.load(weights)
    weights_resampled = ants.resample_image_to_target(
        image=ants.from_nibabel(weights_raw_nii),
        target=ants.from_nibabel(nii_clean),
        interp_type="lanczosWindowedSinc",
        imagetype=2,
    )
    del weights_raw_nii
    weights_nii: nb.Nifti1Image = ants.to_nibabel(weights_resampled)
    del weights_resampled

    signal = []
    for t in range(n_tr):
        signal.append(
            np.vdot(weights_nii.get_fdata(), nii_clean.get_fdata()[:, :, :, t])
        )

    return pd.DataFrame.from_dict({"tr": range(n_tr), "signal": signal}).assign(
        img=utils.img_stem(img),
        confounds="+".join([str(x) for x in confounds.columns.values]),
    )


@prefect.task(cache_key_fn=task_input_hash)
def get_files(sub: Path, space: str) -> frozenset[CuffFiles]:
    out = set()
    subgroup = re.search(r"(?<=sub-)\d{5}", str(sub))
    if subgroup is None:
        raise ValueError(f"{sub=} doesn't look like a bids sub directory")
    else:
        s = subgroup.group(0)
    for ses in sub.glob("ses*"):
        sesgroup = re.search(r"(?<=ses-)\w{2}", str(ses))
        if sesgroup is None:
            raise ValueError(f"{ses=} doesn't look like a bids ses directory")
        else:
            e = sesgroup.group(0)
        func = ses / "func"
        for run in ["1", "2"]:
            bold = (
                func
                / f"sub-{s}_ses-{e}_task-cuff_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
            )
            probseg = frozenset(ses.glob(f"anat/*{space}*probseg*"))
            confounds = (
                func
                / f"sub-{s}_ses-{e}_task-cuff_run-{run}_desc-confounds_timeseries.tsv"
            )
            boldref = (
                func
                / f"sub-{s}_ses-{e}_task-cuff_run-{run}_space-{space}_boldref.nii.gz"
            )
            mask = (
                func
                / f"sub-{s}_ses-{e}_task-cuff_run-{run}_space-{space}_desc-brain_mask.nii.gz"
            )
            if (
                bold.exists()
                and boldref.exists()
                and confounds.exists()
                and mask.exists()
                and all([x.exists() for x in probseg])
            ):
                out.add(
                    CuffFiles(
                        bold=bold,
                        boldref=boldref,
                        probseg=probseg,
                        confounds=confounds,
                        mask=mask,
                        stem=utils.img_stem(bold),
                    )
                )

    return frozenset(out)


@prefect.flow
def connectivity_flow(
    subdirs: frozenset[Path],
    out: Path,
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_seconds: float = 15,
    detrend: bool = True,
    space: str = "MNI152NLin2009cAsym",
) -> None:

    nps = utils.get_nps_mask()

    for subdir in subdirs:
        # not submitting to enable acess of individual parts
        for file in get_files(sub=subdir, space=space):
            acompcor = compcor.do_compcor.submit(
                out / "acompcor" / f"img={file.stem}/part-0.parquet",
                img=file.bold,
                boldref=file.boldref,
                probseg=file.probseg,
                high_pass=high_pass,
                low_pass=low_pass,
                n_non_steady_state_seconds=n_non_steady_state_seconds,
                detrend=detrend,
            )

            final_confounds = task_utils.update_confounds.submit(
                out / "confounds" / f"img={file.stem}/part-0.parquet",
                acompcor_file=acompcor,  # type: ignore
                confounds=file.confounds,
            )
            # apply nps weights
            apply_weights.submit(
                out / "voxelwise_connectivity" / f"img={file.stem}/part-0.parquet",
                img=file.bold,
                confounds_file=final_confounds,  # type: ignore
                weights=nps,
                high_pass=high_pass,
                low_pass=low_pass,
                detrend=detrend,
                brain_mask=file.mask,
            )

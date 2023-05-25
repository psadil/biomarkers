from pathlib import Path

import nibabel as nb
from nibabel import processing

import prefect


from biomarkers.flows import cuff

from biomarkers.task import imgs
from biomarkers import utils


@prefect.task
@utils.cache_nii
def get_slice(img: Path, tr: int) -> nb.Nifti1Image:
    return nb.load(img).slicer[:, :, :, tr]


@prefect.task
@utils.cache_nii
def resample_probseg(img: Path, target_file: Path) -> nb.Nifti1Image:
    # PV maps from FAST
    nii = nb.load(img)
    target = nb.load(target_file)

    # Resample probseg maps to BOLD resolution
    # assume already in matching space
    out: nb.Nifti1Image = processing.resample_from_to(from_img=nii, to_vox_map=target)

    return out


@prefect.flow
def debias_flow(
    subdirs: frozenset[Path],
    out: Path,
    space: str = "MNI152NLin2009cAsym",
) -> None:
    for subdir in subdirs:
        for file in cuff.get_imgs.submit(sub=subdir, space=space).result():
            debiased = resample_probseg.submit(
                out / "probseg" / "mask=WM" / f"{file.stem}.nii.gz",
                img=[x for x in file.probseg if "WM" in x.stem][0],
                target_file=file.boldref,
            )
            get_slice.submit(
                out / "biased-slice" / f"{file.stem}.nii.gz",
                img=file.bold,
                tr=0,
            )
            for s, b, c in zip([3, 3], [100, None], [1000, None]):
                debiased = imgs.correct_bias.submit(
                    out / f"debiased_s-{s}_b-{b}_c-{c}" / f"{file.stem}.nii.gz",
                    img=file.bold,
                    mask=file.mask,
                    s=s,
                    b=b,
                    c=c,
                )

                get_slice.submit(
                    out / f"debiased-slice_s-{s}_b-{b}_c-{c}" / f"{file.stem}.nii.gz",
                    img=debiased,  # type: ignore
                    tr=0,
                )

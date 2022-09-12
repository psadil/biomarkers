from __future__ import annotations

import os
import pathlib
import string
from importlib import resources

import numpy as np
import numpy.ma as ma
import pandas as pd

import nibabel as nib

from nilearn import masking

from nipype.interfaces.base import (
    TraitedSpec,
    ImageFile,
    Directory,
    SimpleInterface,
    File,
)

from nipype.interfaces.fsl.base import FSLCommandInputSpec, FSLCommand, Info


class FEAT2InputSpec(FSLCommandInputSpec):
    in_file = ImageFile(
        exists=True,
        manditory=True,
        resolve=True,
        desc="image to process",
    )
    anat_file = ImageFile(
        exists=True,
        copyfile=True,
        manditory=True,
        resolve=True,
        desc="image to process",
    )
    brain_file = ImageFile(
        exists=True,
        copyfile=True,
        manditory=True,
        resolve=True,
        desc="image to process",
    )


class FEAT2OutputSpec(TraitedSpec):
    outputdir = Directory(exists=True)


class FEAT2(FSLCommand):
    """Use FEAT to preprocess"""

    _cmd = "feat feat.fsf"
    input_spec = FEAT2InputSpec
    output_spec = FEAT2OutputSpec
    _fsf_file = pathlib.Path("feat.fsf")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_interface(self, runtime):
        pathlib.Path(self.inputs.brain_file).rename(self.get_brainfile())
        self._fsf_file.write_text(self.get_fsf())
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["outputdir"] = os.path.abspath(self.get_outputdir())
        return outputs

    def get_fsf(self) -> str:
        return string.Template(
            resources.read_text("biomarkers.data", "fsf.template")
        ).substitute(
            {
                "in_file": self.inputs.in_file,
                "highres_file": self.get_brainfile(),
                "regstandard": Info.standard_image("MNI152_T1_2mm_brain.nii.gz"),
                "outputdir": self.get_outputstem(),
                "repetition_time": nib.load(self.inputs.in_file).header["pixdim"][4],
            }
        )

    def get_brainfile(self) -> pathlib.Path:
        t1w_stem = (
            pathlib.Path(self.inputs.anat_file)
            .name.removesuffix(".gz")
            .removesuffix(".nii")
        )
        return pathlib.Path(f"{t1w_stem}_brain.nii.gz").absolute()

    def get_outputstem(self) -> str:
        return (
            pathlib.Path(self.inputs.in_file)
            .name.removesuffix(".gz")
            .removesuffix(".nii")
        )

    def get_outputdir(self) -> str:
        return f"{self.get_outputstem()}.feat"


class FEATImgInputSpec(TraitedSpec):
    feat_dir = Directory(
        exists=True,
        manditory=True,
        resolve=True,
        desc="directory of FEAT analysis",
    )


class FEATImgOutputSpec(TraitedSpec):
    filtered_func_data = ImageFile(desc="image file from input")
    example_func2standard_warp = ImageFile(desc="for use with applywarp")


class FEATImg(SimpleInterface):

    input_spec = FEATImgInputSpec
    output_spec = FEATImgOutputSpec

    def _run_interface(self, runtime):
        self._results["filtered_func_data"] = (
            pathlib.Path(self.inputs.feat_dir) / "filtered_func_data.nii.gz"
        )
        self._results["example_func2standard_warp"] = (
            pathlib.Path(self.inputs.feat_dir)
            / "reg"
            / "example_func2standard_warp.nii.gz"
        )
        return runtime


class ConfoundsInputSpec(TraitedSpec):
    feat_dir = Directory(
        exists=True,
        manditory=True,
        resolve=True,
        desc="directory of FEAT analysis",
    )
    in_file = ImageFile(
        exists=True,
        manditory=True,
        copyfile=False,
        resolve=True,
        desc="image to process",
    )


class ConfoundsOutputSpec(TraitedSpec):
    confounds = File(desc="tsv of counfounds")


class Confounds(SimpleInterface):
    """_summary_

    Several sources of noise were removed through linear regression.
    These included the six parameters obtained by rigid body correction
    of head motion, the whole-brain signal averaged over all voxels of the
    brain, signal from a ventricular ROI and signal from a region centered in
    the white matter.

    Args:
        SimpleInterface (_type_): _description_

    Returns:
        _type_: _description_
    """

    input_spec = ConfoundsInputSpec
    output_spec = ConfoundsOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)

    def _run_interface(self, runtime):
        # NOTE: none of these are in the standard space

        feat_path = pathlib.Path(self.inputs.feat_dir)
        filtered_func_data = nib.load(
            feat_path / "filtered_func_data.nii.gz"
        ).get_fdata()
        mc = pd.read_csv(
            feat_path / "mc" / "prefiltered_func_data_mcf.par",
            names=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
            delim_whitespace=True,
        )
        nii = nib.load(self.inputs.in_file)

        ventricles = masking.apply_mask(
            nii, mask_img=Info.standard_image("MNI152_T1_2mm_VentricleMask.nii.gz")
        )
        avg = pd.DataFrame.from_dict(
            {
                "ventricles": np.mean(
                    ventricles,
                    axis=1,
                ),
            }
        )
        # current implementation does not put in standard space
        # wm = self.mask(
        #     feat_path / "reg" / "example_func2highres_fast_wmseg.nii.gz",
        #     filtered_func_data,
        # )
        # ventricles = self.mask(
        #     Info.standard_image("MNI152_T1_2mm_VentricleMask.nii.gz"), nii
        # )
        # brain = self.mask(Info.standard_image("MNI152_T1_2mm_brain.nii.gz"), nii)

        # avg = pd.DataFrame.from_dict(
        #     {
        #         "brain": np.mean(
        #             brain,
        #             axis=(0, 1, 2),
        #         ),
        #         # "wm": np.mean(
        #         #     wm,
        #         #     axis=(0, 1, 2),
        #         # ),
        #         "ventricles": np.mean(
        #             ventricles,
        #             axis=(0, 1, 2),
        #         ),
        #     }
        # )
        self._results["confounds"] = pathlib.Path("confounds.tsv").absolute()
        confounds = pd.concat([mc, avg], axis=1)
        confounds.to_csv(self._results["confounds"], sep="\t", index=False)
        return runtime

    @staticmethod
    def mask(mask_file: pathlib.Path, functional: np.ndarray) -> ma.MaskedArray:
        mask = nib.load(mask_file).get_fdata() < 1

        return ma.stack(
            [
                ma.masked_where(mask, functional.take(i, axis=3))
                for i in range(functional.shape[3])
            ],
            axis=3,
        )

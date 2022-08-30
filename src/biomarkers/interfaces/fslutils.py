from __future__ import annotations

import os
import pathlib
import string
from importlib import resources

import nibabel as nib

from nipype.interfaces.base import (
    TraitedSpec,
    ImageFile,
    File,
    Directory,
    SimpleInterface,
)

from nipype.interfaces.fsl.base import FSLCommandInputSpec, FSLCommand


class FEAT2InputSpec(FSLCommandInputSpec):
    in_file = ImageFile(
        exists=True,
        manditory=True,
        copyfile=False,
        resolve=True,
        desc="image to process",
    )
    fsf_file = File(
        exists=False,
        mandatory=False,
        argstr="%s",
        position=0,
        desc="dummy input. do no specify. will be ignored",
    )


class FEAT2OutputSpec(TraitedSpec):
    feat_dir = Directory()


class FEAT2(FSLCommand):
    """Use FEAT to preprocess"""

    _cmd = "feat"
    input_spec = FEAT2InputSpec
    output_spec = FEAT2OutputSpec

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.fsf_file = "feat.fsf"

    def _run_interface(self, runtime):
        fsf_file = pathlib.Path(self.inputs.fsf_file)
        fsf_file.write_text(self.get_fsf(self.inputs.in_file, feat_dir=self.feat_dir))
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["feat_dir"] = os.path.abspath(self.feat_dir)
        return outputs

    @staticmethod
    def get_fsf(in_file: str, feat_dir: str = "feat") -> str:
        return string.Template(
            resources.read_text("biomarkers.data", "fsf.template")
        ).substitute(
            {
                "in_file": in_file,
                "feat_dir": feat_dir,
                "repetition_time": nib.load(in_file).header["pixdim"][4],
            }
        )

    @property
    def feat_dir(self) -> str:
        return os.path.basename(self.inputs.in_file)


class FSFInputSpec(TraitedSpec):
    in_file = ImageFile(
        exists=True,
        manditory=True,
        copyfile=False,
        resolve=True,
        desc="image to process",
    )


class FSFOutputSpec(TraitedSpec):
    fsf_file = File(desc="fsf_file reader for FEAT")


class FSF(SimpleInterface):
    input_spec = FSFInputSpec
    output_spec = FSFOutputSpec

    def _run_interface(self, runtime):
        fsf_file = pathlib.Path("feat.fsf").absolute()
        fsf_file.write_text(self.get_fsf(self.inputs.in_file))
        self._results["fsf_file"] = fsf_file
        return runtime

    @staticmethod
    def get_fsf(in_file: str, feat_dir: str = "feat") -> str:
        return string.Template(
            resources.read_text("biomarkers.data", "fsf.template")
        ).substitute({"in_file": in_file, "feat_dir": feat_dir})


class FEATImgInputSpec(TraitedSpec):
    feat_dir = Directory(
        exists=True,
        manditory=True,
        resolve=True,
        desc="directory of FEAT analysis",
    )


class FEATImgOutputSpec(TraitedSpec):
    out_file = ImageFile(desc="image file from input")


class FEATImg(SimpleInterface):

    input_spec = FEATImgInputSpec
    output_spec = FEATImgOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = (
            pathlib.Path(self.inputs.feat_dir) / "filtered_func_data.nii.gz"
        )
        return runtime

from pathlib import Path

from nipype.interfaces.base import ImageFile, Directory, TraitedSpec
from nipype.interfaces.fsl.base import FSLCommandInputSpec, FSLCommand


class FSLAnatInputSpec(FSLCommandInputSpec):
    in_file = ImageFile(
        exists=True,
        mandatory=True,
        position=-2,
        copyfile=False,
        argstr="-i %s",
        desc="input data file",
    )


class FSLAnatOutputSpec(TraitedSpec):
    anat = Directory(exists=True, resolve=True, desc=".data file")


class FSLAnat(FSLCommand):
    # NOTE: there are several options that may be reasonable to turn off (e.g., --noseg).
    # But be careful! For exmaple, the current fsl_anat performs bias correction, and the bias
    # corrected image is used as input into run_first_all. If segmentation is left on (default)
    # then the correction is refined after segmentation, meaning --noseg affects the subcortical
    # segmentation
    _cmd = "fsl_anat"
    input_spec = FSLAnatInputSpec
    output_spec = FSLAnatOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["anat"] = f"{self._get_infile_stem()}.anat"
        return outputs

    def _get_infile_stem(self):
        return Path(self.inputs.in_file).name.removesuffix(".gz").removesuffix(".nii")

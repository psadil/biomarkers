import os.path as op
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import (
    File, 
    traits, 
    TraitedSpec, 
    OutputMultiPath,
    isdefined
)
from nipype.interfaces.fsl.base import (
    FSLCommandInputSpec, 
    FSLCommand
)

class FIRSTInputSpec(FSLCommandInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        position=-2,
        copyfile=False,
        argstr="-i %s",
        desc="input data file",
    )
    out_file = File(
        "segmented",
        usedefault=True,
        mandatory=True,
        position=-1,
        argstr="-o %s",
        desc="output data file",
        hash_files=False,
    )
    verbose = traits.Bool(argstr="-v", position=1, desc="Use verbose logging.")
    brain_extracted = traits.Bool(
        argstr="-b",
        position=2,
        desc="Input structural image is already brain-extracted",
    )
    no_cleanup = traits.Bool(
        argstr="-d",
        position=3,
        desc="Input structural image is already brain-extracted",
    )
    method = traits.Enum(
        "auto",
        "fast",
        "none",
        xor=["method_as_numerical_threshold"],
        argstr="-m %s",
        position=4,
        usedefault=True,
        desc=(
            "Method must be one of auto, fast, none, or it can be entered "
            "using the 'method_as_numerical_threshold' input"
        ),
    )
    method_as_numerical_threshold = traits.Float(
        argstr="-m %.4f",
        position=4,
        desc=(
            "Specify a numerical threshold value or use the 'method' input "
            "to choose auto, fast, or none"
        ),
    )
    list_of_specific_structures = traits.List(
        traits.Str,
        argstr="-s %s",
        sep=",",
        position=5,
        minlen=1,
        desc="Runs only on the specified structures (e.g. L_Hipp, R_Hipp"
        "L_Accu, R_Accu, L_Amyg, R_Amyg"
        "L_Caud, R_Caud, L_Pall, R_Pall"
        "L_Puta, R_Puta, L_Thal, R_Thal, BrStem",
    )
    affine_file = File(
        exists=True,
        position=6,
        argstr="-a %s",
        desc=(
            "Affine matrix to use (e.g. img2std.mat) (does not " "re-run registration)"
        ),
    )


class FIRSTOutputSpec(TraitedSpec):
    vtk_surfaces = OutputMultiPath(
        File(exists=True), desc="VTK format meshes for each subcortical region"
    )
    bvars = OutputMultiPath(File(exists=True), desc="bvars for each subcortical region")
    original_segmentations = File(
        exists=True,
        desc=(
            "3D image file containing the segmented regions "
            "as integer values. Uses CMA labelling"
        ),
    )
    segmentation_file = File(
        exists=True,
        desc=("4D image file containing a single volume per " "segmented region"),
    )
    to_std_sub_nii = File(
        exists=True,
        desc=("for checking results")
    )
    to_std_sub_mat = File(
        exists=True,
        desc=("matrix")
    )


class FIRST(FSLCommand):
    """FSL run_first_all wrapper for segmentation of subcortical volumes
    http://www.fmrib.ox.ac.uk/fsl/first/index.html
    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> first = fsl.FIRST()
    >>> first.inputs.in_file = 'structural.nii'
    >>> first.inputs.out_file = 'segmented.nii'
    >>> res = first.run() #doctest: +SKIP
    """

    _cmd = "run_first_all"
    input_spec = FIRSTInputSpec
    output_spec = FIRSTOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if isdefined(self.inputs.list_of_specific_structures):
            structures = self.inputs.list_of_specific_structures
        else:
            structures = [
                "L_Hipp",
                "R_Hipp",
                "L_Accu",
                "R_Accu",
                "L_Amyg",
                "R_Amyg",
                "L_Caud",
                "R_Caud",
                "L_Pall",
                "R_Pall",
                "L_Puta",
                "R_Puta",
                "L_Thal",
                "R_Thal",
                "BrStem",
            ]
        outputs["original_segmentations"] = self._gen_fname("original_segmentations")
        outputs["segmentation_file"] = self._gen_fname("segmentation_file")
        outputs["to_std_sub_nii"] = self._gen_fname("to_std_sub_nii")
        outputs["to_std_sub_mat"] = self._gen_fname("to_std_sub_mat")
        outputs["vtk_surfaces"] = self._gen_mesh_names("vtk", structures)
        outputs["bvars"] = self._gen_mesh_names("bvars", structures)
        return outputs

    def _gen_fname(self, basename):
        _, outname, _ = split_filename(self.inputs.out_file)

        method = "none"
        if isdefined(self.inputs.method) and self.inputs.method != "none":
            method = "fast"
            if self.inputs.list_of_specific_structures and self.inputs.method == "auto":
                method = "none"

        if isdefined(self.inputs.method_as_numerical_threshold):
            thres = "%.4f" % self.inputs.method_as_numerical_threshold
            method = thres.replace(".", "")

        if basename == "original_segmentations":
            return op.abspath(f"{outname}_all_{method}_origsegs.nii.gz")
        if basename == "segmentation_file":
            return op.abspath(f"{outname}_all_{method}_firstseg.nii.gz")
        if basename == "to_std_sub_nii":
            return op.abspath(f"{outname}_to_std_sub.nii.gz")
        if basename == "to_std_sub_mat":
            return op.abspath(f"{outname}_to_std_sub.mat")

        return None

    def _gen_mesh_names(self, suffix, structures):
        _, prefix, _ = split_filename(self.inputs.out_file)
        if suffix in ["vtk", "bvars"]:
            out = []
            for struct in structures:
                out.append(op.abspath(f"{prefix}-{struct}_first.{suffix}"))
        else:
            out = None

        return out

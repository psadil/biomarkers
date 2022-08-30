import pathlib

import numpy as np

from nipype.interfaces.base import (
    TraitedSpec,
    ImageFile,
    File,
    DynamicTraitedSpec,
    BaseInterface,
)

from nilearn.maskers import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure


class FEATConInputSpec(TraitedSpec):
    in_file = ImageFile(
        exists=True,
        manditory=True,
        copyfile=False,
        resolve=True,
        desc="image to process",
    )


class FEATConOutputSpec(DynamicTraitedSpec):
    correlation_matrix = File(desc="correlation matrix")


class FEATCon(BaseInterface):

    input_spec = FEATConInputSpec
    output_spec = FEATConOutputSpec

    def _run_interface(self, runtime):

        # labels = ["Medial Prefrontal Cortex", "Right Nucleus Accumbens"]
        coords = [(2, 52, -2), (10, 12, -8)]
        radius = 10

        masker = NiftiSpheresMasker(coords, radius=radius)

        # Additionally, we pass confound information to ensure our extracted
        # signal is cleaned from confounds.
        time_series = masker.fit_transform(self.inputs.in_file)

        connectivity_measure = ConnectivityMeasure(kind="correlation")
        correlation_matrix = connectivity_measure.fit_transform([time_series])

        np.savetxt(
            fname=self._get_outfile(),
            X=correlation_matrix[0],
            header=" ".join(x for x in coords),
        )

        return runtime

    def _list_outputs(self) -> dict:
        outputs = self.output_spec().get()
        outputs["correlation_matrix"] = self._get_outfile()
        return outputs

    def _get_outfile(self) -> str:
        return f"{self._get_infile_stem()}.txt"

    def _get_infile_stem(self):
        return (
            pathlib.Path(self.inputs.in_file)
            .name.removesuffix(".gz")
            .removesuffix(".nii")
        )

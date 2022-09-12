import os
import pathlib

import pandas as pd

from nipype.interfaces.base import (
    TraitedSpec,
    ImageFile,
    File,
    BaseInterface,
)

from nilearn.maskers import NiftiSpheresMasker


class FEATConInputSpec(TraitedSpec):
    in_file = ImageFile(
        exists=True,
        manditory=True,
        copyfile=False,
        resolve=True,
        desc="image to process",
    )
    confounds = ImageFile(
        exists=True,
        manditory=False,
        copyfile=False,
        resolve=True,
        default=None,
        desc="tsv of confounds",
    )


class FEATConOutputSpec(TraitedSpec):
    correlation_matrix = File(desc="correlation matrix")


class FEATCon(BaseInterface):

    input_spec = FEATConInputSpec
    output_spec = FEATConOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)

    def _run_interface(self, runtime):

        # labels = ["Medial Prefrontal Cortex", "Right Nucleus Accumbens"]
        coords = [(2, 52, -2), (10, 12, -8)]
        radius = 10

        masker = NiftiSpheresMasker(coords, radius=radius)

        # Additionally, we pass confound information to ensure our extracted
        # signal is cleaned from confounds.
        time_series = masker.fit_transform(self.inputs.in_file)
        d = pd.DataFrame(time_series, columns=coords)
        d.corr().to_csv(self._get_outfile(), sep="\t")

        # connectivity_measure = ConnectivityMeasure(kind="correlation")
        # correlation_matrix = connectivity_measure.fit_transform([time_series])

        # np.savetxt(
        #     fname=self._get_outfile(),
        #     X=correlation_matrix[0],
        #     header=" ".join(str(x) for x in coords),
        # )

        return runtime

    def _list_outputs(self) -> dict:
        outputs = self.output_spec().get()
        outputs["correlation_matrix"] = os.path.abspath(self._get_outfile())
        return outputs

    def _get_outfile(self) -> str:
        return f"{self._get_infile_stem()}.tsv"

    def _get_infile_stem(self):
        return (
            pathlib.Path(self.inputs.in_file)
            .name.removesuffix(".gz")
            .removesuffix(".nii")
        )

import numpy as np

from biomarkers.workflows import rest
from nipype.utils.filemanip import loadpkl

rest_wf = rest.RestWF(base_dir="scratch")
rest_wf.inputs.inputnode.in_file = (
    "data/bids/sub-travel2/ses-NS/func/sub-travel2_ses-NS_task-rest_run-01_bold.nii.gz"
)
rest_wf.inputs.inputnode.highres_file = (
    "data/bids/sub-travel2/ses-NS/anat/sub-travel2_ses-NS_T1w_brain.nii.gz"
)
rest_wf.inputs.inputnode.wholehead_file = (
    "data/bids/sub-travel2/ses-NS/anat/sub-travel2_ses-NS_T1w.nii.gz"
)


rest_wf.config["execution"] = {
    "crashfile_format": "txt",
    "keep_unnecessary_outputs": True,
}

rest_wf.run()

x = loadpkl("scratch/rest/fc/result_fc.pklz")

r = np.loadtxt(x.outputs.correlation_matrix)

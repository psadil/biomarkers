import nipype
from biomarkers.biomarkers import MainWF
from biomarkers.workflows import rest

rest_wf = rest.RestWF(base_dir="scratch")
rest_wf.inputs.inputnode.in_file = (
    "data/bids/sub-travel2/ses-NS/func/sub-travel2_ses-NS_task-rest_run-01_bold.nii.gz"
)

assert issubclass(MainWF, nipype.Workflow)

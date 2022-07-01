import pathlib
from biomarkers.workflows import anat

def test_anat_wf():
    anat_wf = anat.init_anat_wf(
        img=[pathlib.Path("tests/sub-travel2_ses-NS_T1w.nii.gz").absolute()],
        output_dir=str(pathlib.Path("out").absolute()),
        work_dir=str(pathlib.Path("work").absolute())
        )    

    anat_wf.run()


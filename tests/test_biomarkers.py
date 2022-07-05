import nipype
from biomarkers.biomarkers import MainWF

assert issubclass(MainWF, nipype.Workflow)

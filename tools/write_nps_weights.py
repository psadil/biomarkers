import nibabel as nb

x = nb.load("2013_Wager_NEJM_NPS/weights_NSF_grouppred_cvpcr.img")

y = nb.Nifti1Image(dataobj=x.dataobj, affine=x.affine, header=x.header)

y.to_filename("../src/biomarkers/data/weights_NSF_grouppred_cvpcr.nii.gz")

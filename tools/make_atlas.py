import numpy as np
import pandas as pd
import nibabel as nb

# from scipy import interpolate
from nilearn import datasets

rng = np.random.default_rng(seed=1234)

# cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
# sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-2mm")
aal = datasets.fetch_atlas_aal()

aal_nii = nb.load(aal.maps)

aal_nii.to_filename("aal.nii.gz")

aal_a = np.asanyarray(aal_nii.dataobj)

indices = [int(x) for x in aal.indices]
vols = dict()
for l, label in enumerate(aal.labels):
    vols.update({label: np.sum(aal_a == indices[l])})
d = (
    pd.DataFrame.from_dict(vols, orient="index", columns=["n_voxels"])
    .reset_index()
    .rename(columns={"index": "roi"})
    .assign(src="aal", i=indices)
)

# sub_a = np.asanyarray(sub.maps.dataobj)
# cort_a = np.asanyarray(cort.maps.dataobj)

# vols = dict()
# for l, label in enumerate(sub.labels[1:], 1):
#     vols.update({label: np.sum(sub_a == l)})
# s = (
#     pd.DataFrame.from_dict(vols, orient="index", columns=["n_voxels"])
#     .reset_index()
#     .rename(columns={"index": "roi"})
#     .reset_index()
#     .assign(src="subcortical")
# )

# vols = dict()
# for l, label in enumerate(cort.labels[1:], 1):
#     vols.update({label: np.sum(cort_a == l)})
# c = (
#     pd.DataFrame.from_dict(vols, orient="index", columns=["n_voxels"])
#     .reset_index()
#     .rename(columns={"index": "roi"})
#     .reset_index()
#     .assign(src="cortex")
# )
# d = pd.concat([s, c])
N_voxels = d["n_voxels"].sum()
N_nodes = 500

d["prop"] = d["n_voxels"] / N_voxels
d["n_nodes"] = np.round(d["prop"] * N_nodes).astype(int)

i = 1
out = np.zeros_like(aal_a)
cogs = []
for roi in d.itertuples():
    if roi.n_nodes == 0:
        print(f"{roi=}")
        continue
    img = aal_a == roi.i
    lin_inds = np.flatnonzero(img)
    cogs = rng.choice(lin_inds, size=roi.n_nodes, replace=False)
    cogs_xyz = np.array(np.unravel_index(cogs, img.shape)).T
    for x, y, z in zip(*np.unravel_index(lin_inds, img.shape)):
        closest_xyz = np.sqrt(
            np.sum((cogs_xyz - np.array([x, y, z])) ** 2, axis=1)
        ).argmin()
        out[x, y, z] = i + closest_xyz
    i += len(cogs)

out_nii = nb.Nifti1Image(out, affine=aal_nii.affine, header=aal_nii.header)
out_nii.to_filename("atlas.nii.gz")


# X, Y, Z = np.meshgrid(
#     np.arange(0, sub_a.shape[0]),
#     np.arange(0, sub_a.shape[1]),
#     np.arange(0, sub_a.shape[2]),
#     indexing="ij",
# )

# i = 1
# out = np.zeros_like(sub_a)
# cogs = []
# for roi in d.itertuples():
#     if roi.src == "cortex":
#         img = sub_a == roi.index
#     else:
#         img = cort_a == roi.index
#     if roi.n_nodes < 1 or img.sum() < 1:
#         continue
#     lin_inds = np.flatnonzero(img)
#     X, Y, Z = np.unravel_index(lin_inds, img.shape)
#     cogs = np.random.choice(lin_inds, size=roi.n_nodes, replace=False)
#     cogs_x, cogs_y, cogs_z = np.unravel_index(cogs, img.shape)
#     for x, y, z in zip(X, Y, Z):
#         closest_xyz = np.sqrt(
#             np.sum(np.asanyarray((cogs_x - x, cogs_y - y, cogs_z - z)) ** 2)
#         ).argmin()
#         out[x, y, z] = i + closest_xyz
#     i += len(cogs)

# cogs = np.unravel_index(
#     np.random.choice(np.flatnonzero(img), size=roi.n_nodes, replace=False),
#     img.shape,
# )
# interp = interpolate.NearestNDInterpolator(
#     np.asanyarray(cogs).T, np.arange(i, i + roi.n_nodes)
# )
# i += roi.n_nodes
# q = interp(X, Y, Z)
# q[img == False] = 0
# out += q

# cog_xyz = np.unravel_index(np.concatenate(cogs), cort_a.shape)

# interp = interpolate.NearestNDInterpolator(
#     np.asanyarray(cog_xyz).T, np.arange(1, 1 + len(cog_xyz[0]))
# )
# out = interp(X, Y, Z)
# mask_nii = nb.load(
#     "/mnt/60C4F60CC4F5E3E6/Users/psadi/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
# )
# mask = np.asanyarray(mask_nii.dataobj)
# out[mask == 0] = 0
# # interp = interpolate.interpn(
# #     points =     np.arange(0, sub_a.shape[0]),
# #     np.arange(0, sub_a.shape[1]),
# #     np.arange(0, sub_a.shape[2]),
# # method="nearest")

# out_nii = nb.Nifti1Image(out, affine=cort.maps.affine, header=cort.maps.header)
# out_nii.to_filename("atlas.nii.gz")

# sub.maps.to_filename("sub.nii.gz")
# cort.maps.to_filename("cort.nii.gz")

import ants
from nilearn import masking
import nibabel as nb


gray1 = ants.image_read("MNI152_T1_1mm_gray.nii.gz")
gray6 = ants.resample_image(gray1, (6, 6, 6), False, 1)
ants.image_write(gray6, "MNI152_T1_6mm_gray.nii.gz")

"""
- read FAST seg of MNI (1mm) (from R package MNITemplate)
- write out binary nifti of gray matter (MNI152_T1_1mm_gray)
- downsample to 6mm^3, nearest neighbor (ants.resample_image(gray1, (6, 6, 6), False, 1))
- use this as mask

- for each fmri
    - clean signals
    - downsample to 6mm, bspline (ants.resample_image(gray1, (6, 6, 6), False, 4))
    - mask
    - calculate connectivity
"""

m6 = ants.image_read("MNI152_T1_6mm_gray.nii.gz")


zz = ants.image_read(
    "sub-20064_ses-V3_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)

zz6 = ants.resample_image_to_target(
    image=zz,
    target=m6,
    interp_type="lanczosWindowedSinc",
    imagetype=3,
)

zz6_nii = ants.to_nibabel(zz6)
zz6_nii.to_filename("zz6-ants.nii.gz")
test = masking.apply_mask(zz6_nii, "MNI152_T1_6mm_gray.nii.gz")

from fsl.utils.image import resample
from fsl.data import image

mm = nb.load(
    "sub-20064_ses-V3_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)

x = resample.resampleToReference(
    image.Image(
        "sub-20064_ses-V3_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    ),
    image.Image("MNI152_T1_6mm_gray.nii.gz"),
    order=3,
)
nb.Nifti1Image(x[0], affine=x[1]).to_filename("resampled-corner.nii.gz")
test = masking.apply_mask(
    nb.Nifti1Image(x[0], affine=x[1]), "MNI152_T1_6mm_gray.nii.gz"
)

x = resample.resampleToReference(image.Image(mm), image.Image(m6))

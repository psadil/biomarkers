import re

import pandas as pd
from scipy import io

mat = io.loadmat("cluster_Fan_Net_r279.mat")

pain_clusters = {0: pd.NA}
for p, pain_cluster in enumerate(
    mat["cluster_Fan_Net"]["pain_cluster_names"][0][0], start=1
):
    pain_clusters.update({p: re.findall(r"[a-zA-Z]+\w*", pain_cluster[0][0])[0]})

full_names = {}
for n, name in enumerate(mat["cluster_Fan_Net"]["full_names"][0][0], start=1):
    full_names.update({n: name[0][0]})


cluster_names = {}
for n, name in enumerate(mat["cluster_Fan_Net"]["cluster_names"][0][0], start=1):
    cluster_names.update({n: name[0][0]})


d = pd.DataFrame(
    mat["cluster_Fan_Net"]["dat"][0][0],
    columns=[
        "original",
        "manual_buckner",
        "manual_bucker2",
        "cluster",
        "pain_cluster",
        "anat_cluster_names",
        "laterality",
        "brainnetome",
        "w_brainstem_cerebellum",
        "update",
        "update2",
        "update3",
        "update9",
    ],
)

d["cluster"] = [cluster_names[x] for x in d["cluster"]]
d["pain_cluster"] = [pain_clusters[x] for x in d["pain_cluster"]]
d["name"] = full_names

d[["name", "cluster", "pain_cluster"]].assign(value=range(1, 280)).to_csv(
    "../src/biomarkers/data/fan_atlas.csv", index=False
)

from nilearn import maskers

m = maskers.NiftiLabelsMasker(
    labels_img="../src/biomarkers/data/Fan_et_al_atlas_r279_MNI_2mm.nii.gz",
    resampling_target="data",
)
time_series = m.fit_transform(
    imgs="sub-20064_ses-V3_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)

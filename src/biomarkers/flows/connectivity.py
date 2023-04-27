from pathlib import Path
import re

import numpy as np

import nibabel as nb
import pandas as pd

from sklearn import covariance

import pydantic
from pydantic.dataclasses import dataclass

from nilearn import maskers
from nilearn.connectome import ConnectivityMeasure
from nilearn import image
from nilearn import masking


import networkx as nx

# import statsmodels as sm
# import patsy

import prefect

# from prefect.task_runners import SequentialTaskRunner


from .. import utils
from ..task import compcor
from ..task import utils as task_utils


# TODO: remove 8 nodes from ower2011 atlas that are in the cerebellum


@dataclass(frozen=True)
class Coordinate:
    label: str
    seed: tuple[int, int, int]


@dataclass(frozen=True)
class ConnectivityFiles:
    bold: pydantic.FilePath
    boldref: pydantic.FilePath
    probseg: frozenset[pydantic.FilePath]
    confounds: pydantic.FilePath
    mask: pydantic.FilePath
    stem: str


def df_to_coordinates(dataframe: pd.DataFrame) -> frozenset[Coordinate]:
    coordinates = set()
    for row in dataframe.itertuples():
        coordinates.add(Coordinate(label=row.label, seed=(row.x, row.y, row.z)))

    return frozenset(coordinates)


@prefect.task
def get_baliki_coordinates() -> frozenset[Coordinate]:
    return frozenset(
        {
            Coordinate(label="mPFC", seed=(2, 52, -2)),
            Coordinate(label="rNAc", seed=(10, 12, -8)),
            Coordinate(label="rInsula", seed=(40, -6, -2)),
            Coordinate(label="S1/M1", seed=(-32, -34, 66)),
        },
    )


def get_power_coordinates() -> frozenset[Coordinate]:
    from nilearn import datasets

    rois: pd.DataFrame = datasets.fetch_coords_power_2011(legacy_format=False).rois
    rois.query("not roi in [127, 183, 184, 185, 243, 244, 245, 246]", inplace=True)
    rois.rename(columns={"roi": "label"}, inplace=True)
    return df_to_coordinates(rois)


@prefect.task
@utils.cache_dataframe
def spheres_connectivity(
    img: Path,
    confounds_file: Path,
    coordinates: frozenset[Coordinate],
    radius: int = 5,  # " ... defined as 10-mm spheres centered ..."
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:

    """
    for confounds,
    - Friston24,
    - top 5 principal components
    """
    confounds = pd.read_parquet(confounds_file)

    n_tr = confounds.shape[0]
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, -n_tr:]
    masker = maskers.NiftiSpheresMasker(
        seeds=[x.seed for x in coordinates],
        radius=radius,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        standardize=False,
        standardize_confounds=True,
        detrend=detrend,
    )
    # confounds are already sliced
    time_series = masker.fit_transform(nii, confounds=confounds)
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=covariance.EmpiricalCovariance(store_precision=False),  # type: ignore
        kind="correlation",
    )
    correlation_matrix = connectivity_measure.fit_transform([time_series]).squeeze()  # type: ignore
    df = utils._mat_to_df(correlation_matrix, [x.label for x in coordinates])
    df["connectivity"] = np.arctanh(df["connectivity"])
    return df


@prefect.task
@utils.cache_dataframe
def get_labels_connectivity(
    img: Path,
    confounds_file: Path,
    labels_img: Path,
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:

    """
    for confounds,
    - Friston24,
    - top 5 principal components
    """
    confounds = pd.read_parquet(confounds_file)

    n_tr = confounds.shape[0]
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, -n_tr:]
    masker = maskers.NiftiLabelsMasker(
        labels_img=labels_img,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        standardize=False,
        standardize_confounds=False,
        detrend=detrend,
        resampling_target="data",
    )
    # confounds are already sliced
    time_series = masker.fit_transform(nii, confounds=confounds)
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=covariance.EmpiricalCovariance(store_precision=False),  # type: ignore
        kind="correlation",
    )
    correlation_matrix: np.ndarray = connectivity_measure.fit_transform(
        [time_series]
    ).squeeze()  # type: ignore
    df = utils._mat_to_df(
        correlation_matrix, [str(x + 1) for x in range(correlation_matrix.shape[0])]
    ).assign(
        img=utils.img_stem(img),
        confounds="+".join([str(x) for x in confounds.columns.values]),
    )
    return df


@prefect.task
@utils.cache_dataframe
def get_gray_connectivity(
    img: Path,
    confounds_file: Path,
    brain_mask: Path | None = None,
    mask_img: Path = utils.get_mni6gray_mask(),
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:
    """
    for each fmri
    - clean signals
    - downsample to 6mm, bspline
    - mask
    - calculate connectivity

    Note: Mansour et al. 2016 used raw correlations, not atanh transform

    Returns:
        _type_: _description_
    """
    import ants

    confounds = pd.read_parquet(confounds_file)
    n_tr = confounds.shape[0]
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, -n_tr:]
    nii_clean: nb.Nifti1Image = image.clean_img(
        imgs=nii,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        confounds=confounds,
        standardize=False,
        detrend=detrend,
        mask_img=brain_mask,
    )  # type: ignore
    del nii

    mask_nii = nb.load(mask_img)
    ants6 = ants.resample_image_to_target(
        image=ants.from_nibabel(nii_clean),
        target=ants.from_nibabel(mask_nii),
        interp_type="lanczosWindowedSinc",
        imagetype=3,
    )
    del nii_clean
    imgs: nb.Nifti1Image = ants.to_nibabel(ants6)
    del ants6
    # n_tr x n_voxels
    X: np.ndarray = masking.apply_mask(imgs=imgs, mask_img=mask_nii)
    del imgs
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=covariance.EmpiricalCovariance(store_precision=False),  # type: ignore
        kind="correlation",
    )
    correlation_matrix = connectivity_measure.fit_transform([X]).squeeze()  # type: ignore
    del X
    df = utils._mat_to_df(
        correlation_matrix, np.flatnonzero(np.asanyarray(mask_nii.dataobj))
    ).assign(
        img=utils.img_stem(img),
        confounds="+".join([str(x) for x in confounds.columns.values]),
    )
    return df


def get_files(sub: Path, space: str) -> frozenset[ConnectivityFiles]:
    out = set()
    subgroup = re.search(r"(?<=sub-)\d{5}", str(sub))
    if subgroup is None:
        raise ValueError(f"{sub=} doesn't look like a bids sub directory")
    else:
        s = subgroup.group(0)
    for ses in sub.glob("ses*"):
        sesgroup = re.search(r"(?<=ses-)\w{2}", str(ses))
        if sesgroup is None:
            raise ValueError(f"{ses=} doesn't look like a bids ses directory")
        else:
            e = sesgroup.group(0)
        func = ses / "func"
        for run in ["1", "2"]:
            bold = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
            )
            boldref = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_space-{space}_boldref.nii.gz"
            )
            probseg = frozenset(ses.glob(f"anat/*{space}*probseg*"))
            confounds = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_desc-confounds_timeseries.tsv"
            )
            mask = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_space-{space}_desc-brain_mask.nii.gz"
            )
            if (
                bold.exists()
                and boldref.exists()
                and confounds.exists()
                and mask.exists()
                and all([x.exists() for x in probseg])
            ):
                out.add(
                    ConnectivityFiles(
                        bold=bold,
                        boldref=boldref,
                        probseg=probseg,
                        confounds=confounds,
                        mask=mask,
                        stem=utils.img_stem(bold),
                    )
                )

    return frozenset(out)


def get_nedges(n_nodes: int, density: float) -> int:
    return np.floor_divide(density * n_nodes * (n_nodes - 1), 2).astype(int)


def df_to_graph(connectivity: pd.DataFrame, link_density: float) -> nx.Graph:
    g = nx.Graph()
    nodes = set(
        connectivity["source"].unique().tolist()
        + connectivity["target"].unique().tolist()
    )
    g.add_nodes_from(nodes)
    connectivity["connectivity"] = connectivity["connectivity"].abs()
    trimmed = connectivity.nlargest(
        n=get_nedges(n_nodes=len(nodes), density=link_density),
        columns="connectivity",
    )
    g.add_edges_from(trimmed[["source", "target"]].itertuples(index=False))
    return g


@prefect.task
@utils.cache_dataframe
def get_degree(
    connectivity: Path, src: Path, link_density: frozenset[float] = frozenset({0.1})
) -> pd.DataFrame:

    df = pd.read_parquet(connectivity)
    degrees = []
    # careful to avoid counting by 0.01
    for density in link_density:
        g = df_to_graph(df, link_density=density)
        degrees.append(
            pd.DataFrame.from_dict(dict(g.degree), orient="index", columns=["degree"])
            .reset_index()
            .rename(columns={"index": "roi"})
            .assign(density=density)
        )

    return pd.concat(degrees, ignore_index=True).assign(src=src.name)


# def get_hub_disruption(degrees: list[pd.DataFrame]) -> pd.DataFrame:
#     d = pd.concat(degrees, ignore_index=True)
#     avgs = d.groupby("roi").agg({"degree": "mean"}).rename(columns={"degree": "avg"})
#     d = d.join(avgs, on="roi")
#     d["degree_centered"] = d["degree"] - d["avg"]

#     # this could be refactored to run in parallel (e.g., convert this task to a flow),
#     # but I'm banking on the fitting being quick enough that it won't matter
#     hub_disruption = []
#     for name, group in d.groupby("src"):
#         y, X = patsy.dmatrices(
#             "degree_centered ~ avg", data=group, return_type="dataframe"
#         )
#         model = sm.OLS(y, X)
#         fit = model.fit()
#         hub_disruption.append(
#             pd.DataFrame(
#                 {
#                     "slope": [fit.params.avg],
#                     "src": [name],
#                 }
#             )
#         )

#     return pd.concat(hub_disruption, ignore_index=True)


# @prefect.flow(task_runner=SequentialTaskRunner)
@prefect.flow
def connectivity_flow(
    subdirs: frozenset[Path],
    out: Path,
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 12,
    detrend: bool = False,
    space: str = "MNI152NLin2009cAsym",
    link_density: frozenset[float] = frozenset({0.1}),
) -> None:
    baliki_coordinates = get_baliki_coordinates.submit()

    for subdir in subdirs:
        # not submitting to enable acess of individual parts
        for file in get_files(sub=subdir, space=space):
            acompcor = compcor.do_compcor.submit(
                out / "acompcor" / f"img={file.stem}/part-0.parquet",
                img=file.bold,
                boldref=file.boldref,
                probseg=file.probseg,
                high_pass=high_pass,
                low_pass=low_pass,
                n_non_steady_state_tr=n_non_steady_state_tr,
                detrend=detrend,
            )

            final_confounds = task_utils.update_confounds.submit(
                out / "confounds" / f"img={file.stem}/part-0.parquet",
                acompcor_file=acompcor,  # type: ignore
                confounds=file.confounds,
                label="WM+CSF",
                n_non_steady_state_tr=n_non_steady_state_tr,
            )

            spheres_connectivity.submit(
                out / "dmn_connectivity" / f"img={file.stem}/part-0.parquet",
                img=file.bold,
                coordinates=baliki_coordinates,  # type: ignore
                confounds_file=final_confounds,  # type: ignore
                high_pass=high_pass,
                low_pass=low_pass,
                detrend=detrend,
            )
            # voxelwise_connectivity = get_gray_connectivity.submit(
            #     out / "voxelwise_connectivity" / f"img={file.stem}/part-0.parquet",
            #     img=file.bold,
            #     confounds_file=final_confounds,  # type: ignore
            #     high_pass=high_pass,
            #     low_pass=low_pass,
            #     detrend=detrend,
            #     brain_mask=file.mask,
            # )
            # get_degree.submit(
            #     out / "degree" / f"img={file.stem}/part-0.parquet",
            #     connectivity=voxelwise_connectivity,  # type: ignore
            #     src=file.bold,
            #     link_density=link_density,
            # )
            get_labels_connectivity.submit(
                out / "labels_connectivity" / f"img={file.stem}/part-0.parquet",
                img=file.bold,
                confounds_file=final_confounds,  # type: ignore
                labels_img=utils.get_fan_atlas_file(),
                high_pass=high_pass,
                low_pass=low_pass,
                detrend=detrend,
            )

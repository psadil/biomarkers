from __future__ import annotations
from pathlib import Path

import click

import prefect

# from .workflows.rest import RestWF
from .flows.fslanat import fslanat_flow
from .flows.cat import cat_flow


@prefect.flow
def _main(
    bids_dir: Path,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    anat: bool = False,
    rest: bool = False,
) -> None:

    if anat:
        t1w = {x for x in bids_dir.glob("*.nii.gz")}
        fslanat_flow(src=t1w, out=output_dir)
        if cat_dir:
            cat_flow(cat_dir=cat_dir, out=output_dir)
    if rest:
        pass


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "bids_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option("--anat", default=False, is_flag=True)
@click.option("--rest", default=False, is_flag=True)
@click.option(
    "--cat-dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option("--anat", default=False, is_flag=True)
@click.option("--rest", default=False, is_flag=True)
def main(
    bids_dir: Path,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    anat: bool = False,
    rest: bool = False,
) -> None:

    _main(
        output_dir=output_dir,
        bids_dir=bids_dir,
        anat=anat,
        rest=rest,
        cat_dir=cat_dir,
    )

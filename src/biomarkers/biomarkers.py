from pathlib import Path

import click

import nipype
import bids

from .nodes import io
from .workflows.anat import AnatWF


# TODO: configure outputs directory (blocking on decision from group)


class MainWF(nipype.Workflow):
    def __init__(self, anat: list[Path], output_dir: Path, **kwargs):
        super().__init__(name="biomarkers", **kwargs)
        inputnode = io.InputNode.from_fields(["in_file"], iterables=[("in_file", anat)])
        datasink = nipype.Node(
            nipype.DataSink(
                base_directory=output_dir,
                regexp_substitutions=[
                    (r"_in_file_.*/", ""),
                ],
            ),
            name="datasink",
            run_without_submitting=True,
        )
        anat_wf = AnatWF()
        self.connect(
            [
                (inputnode, anat_wf, [("in_file", "inputnode.in_file")]),
                (
                    anat_wf,
                    datasink,
                    [("outputnode.volumes", "@volumes"), ("outputnode.anat", "@anat")],
                ),
            ]
        )


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "src",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--base-dir",
    default="work",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--plugin",
    default="Linear",
    type=click.Choice(choices=["Linear", "MultiProc"]),
)
def main(
    src: Path,
    output_dir: Path = Path("out"),
    base_dir: Path = Path("work"),
    plugin: str = "Linear",
) -> None:

    layout = bids.BIDSLayout(root=src)
    anat = layout.get(return_type="file", suffix="T1w", extension="nii.gz")

    wf = MainWF(anat=anat, output_dir=output_dir, base_dir=base_dir)

    wf.run(plugin)

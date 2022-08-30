from __future__ import annotations
from pathlib import Path

import click

import nipype
import bids

from .nodes import io
from .workflows.anat import AnatWF
from .workflows.rest import RestWF


# TODO: configure outputs directory (blocking on decision from group)


class MainWF(nipype.Workflow):
    def __init__(
        self,
        output_dir: Path,
        anat: list[Path] | None = None,
        rest: list[Path] | None = None,
        **kwargs,
    ):
        super().__init__(name="biomarkers", **kwargs)
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
        if anat:
            self._connect_anat(anat, datasink=datasink)
        if rest:
            self._connect_rest(rest, datasink=datasink)

    def _connect_anat(self, anat: list[Path], datasink: nipype.Node) -> MainWF:
        inputnode = io.InputNode.from_fields(
            ["in_file"], iterables=[("in_file", anat)], name="input_anat"
        )
        wf = AnatWF()
        self.connect(
            [
                (inputnode, wf, [("in_file", "inputnode.in_file")]),
                (
                    wf,
                    datasink,
                    [
                        ("outputnode.volumes", "@volumes"),
                        ("outputnode.anat", "@anat"),
                    ],
                ),
            ]
        )
        return self

    def _connect_rest(self, rest: list[Path], datasink: nipype.Node) -> MainWF:
        inputnode = io.InputNode.from_fields(
            ["in_file"], iterables=[("in_file", rest)], name="input_rest"
        )
        wf = RestWF()
        self.connect(
            [
                (inputnode, wf, [("in_file", "inputnode.in_file")]),
                (
                    wf,
                    datasink,
                    [("outputnode.correlation_matrix", "@correlation_matrix")],
                ),
            ]
        )
        return self


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
@click.option("--anat", default=True, is_flag=True)
@click.option("--rest", default=True, is_flag=True)
@click.option(
    "--plugin",
    default="Linear",
    type=click.Choice(choices=["Linear", "MultiProc"]),
)
def main(
    src: Path,
    output_dir: Path = Path("out"),
    base_dir: Path = Path("work"),
    anat: bool = True,
    rest: bool = True,
    plugin: str = "Linear",
) -> None:

    layout = bids.BIDSLayout(root=src)
    kwargs = {}
    if anat:
        kwargs.update(
            {"anat": layout.get(return_type="file", suffix="T1w", extension="nii.gz")}
        )
    if rest:
        kwargs.update(
            {
                "rest": layout.get(
                    return_type="file", suffix="T1w", extension="nii.gz", task="rest"
                )
            }
        )

    wf = MainWF(output_dir=output_dir, base_dir=base_dir, **kwargs)

    wf.run(plugin)

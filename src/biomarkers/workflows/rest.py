from __future__ import annotations

import nipype

from ..interfaces import connectivity
from ..interfaces import fslutils
from ..nodes import io


class RestWF(nipype.Workflow):
    def __init__(self, **kwargs) -> RestWF:
        super().__init__(name="rest", **kwargs)
        inputnode = io.InputNode.from_fields(["in_file"])
        outputnode = io.OutputNode.from_fields(["correlation_matrix"])

        feat = nipype.Node(fslutils.FEAT2(), name="feat")
        featimg = nipype.Node(fslutils.FEATImg(), name="featimg")
        fc = nipype.Node(connectivity.FEATCon(), name="fc")
        self.connect(
            [
                (inputnode, feat, [("in_file", "in_file")]),
                (feat, featimg, [("feat_dir", "feat_dir")]),
                (featimg, fc, [("out_file", "in_file")]),
                (
                    fc,
                    outputnode,
                    [("correlation_matrix", "correlation_matrix")],
                ),
            ]
        )

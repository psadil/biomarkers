from __future__ import annotations
import attrs

import nipype


@attrs.define
class InputNode(nipype.Node):
    @classmethod
    def from_fields(
        cls, fields: list[str], name: str = "inputnode", **kwargs
    ) -> InputNode:
        return nipype.Node(nipype.IdentityInterface(fields=fields), name=name, **kwargs)


@attrs.define
class OutputNode(nipype.Node):
    @classmethod
    def from_fields(cls, fields: list[str], name: str = "outputnode") -> OutputNode:
        return nipype.Node(nipype.IdentityInterface(fields=fields), name=name)

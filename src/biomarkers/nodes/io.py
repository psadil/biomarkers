import attrs

import nipype


@attrs.define
class InputNode(nipype.Node):
    @classmethod
    def from_fields(cls, fields: list[str], **kwargs) -> "InputNode":
        return nipype.Node(
            nipype.IdentityInterface(fields=fields), name="inputnode", **kwargs
        )


@attrs.define
class OutputNode(nipype.Node):
    @classmethod
    def from_fields(cls, fields: list[str]) -> "OutputNode":
        return nipype.Node(nipype.IdentityInterface(fields=fields), name="outputnode")

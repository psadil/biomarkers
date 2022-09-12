from __future__ import annotations
import nipype


class InputNode(nipype.Node):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_fields(
        cls, fields: list[str], name: str = "inputnode", **kwargs
    ) -> InputNode:
        return cls(
            interface=nipype.IdentityInterface(fields=fields), name=name, **kwargs
        )


class OutputNode(nipype.Node):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_fields(cls, fields: list[str], name: str = "outputnode") -> OutputNode:
        return cls(interface=nipype.IdentityInterface(fields=fields), name=name)

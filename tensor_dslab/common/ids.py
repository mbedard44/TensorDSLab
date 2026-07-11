from __future__ import annotations

from dataclasses import dataclass

from tensor_core import Id


@dataclass(frozen=True, slots=True)
class ExampleId(Id):
    pass


@dataclass(frozen=True, slots=True)
class ChannelId(Id):
    pass

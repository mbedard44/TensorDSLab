# pyright: standard
"""Current public orchestration typing contract."""

from typing import assert_type

from tensor_dslab import ReadoutCollection

assert_type(ReadoutCollection, type[ReadoutCollection])

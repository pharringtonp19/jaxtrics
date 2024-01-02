
from typing import NamedTuple
from jaxlib.xla_extension import ArrayImpl

class Data(NamedTuple):
    X: ArrayImpl
    D: ArrayImpl
    Y: ArrayImpl
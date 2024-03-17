
from typing import NamedTuple, Optional
from jaxlib.xla_extension import ArrayImpl

class Data(NamedTuple):
    X: ArrayImpl
    Z: Optional[ArrayImpl] = None
    D: ArrayImpl
    Y: ArrayImpl

    def __new__(cls, X, D, Y, Z=None):
        return super().__new__(cls, X, Z, D, Y)

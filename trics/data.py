
from typing import NamedTuple, Optional
from jaxlib.xla_extension import ArrayImpl

class Data(NamedTuple):
    X: ArrayImpl
    D: ArrayImpl
    Y: ArrayImpl
    Z: Optional[ArrayImpl] = None

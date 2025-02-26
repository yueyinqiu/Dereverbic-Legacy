from typing import TypeAlias
from typing import Literal


StftWindow: TypeAlias = Literal[
    "hann_window", 
    "kaiser_window", 
    "hamming_window", 
    "bartlett_window", 
    "blackman_window"
]

# The metric is modified from: 
# https://github.com/kyungyunlee/fins/blob/main/fins/loss.py
# Please respect the original license

from typing import TypeAlias
from typing import Literal


StftWindow: TypeAlias = Literal[
    "hann_window", 
    "kaiser_window", 
    "hamming_window", 
    "bartlett_window", 
    "blackman_window"
]

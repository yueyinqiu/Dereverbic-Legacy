from typing import NamedTuple
from statictorch import Tensor2d


class DataBatch(NamedTuple):
    rir: Tensor2d
    speech: Tensor2d
    reverb: Tensor2d

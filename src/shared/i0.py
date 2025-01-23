from pathlib import Path
from random import Random
from torch import Tensor
from typing import (
    Any, 
    Iterable, 
    Callable, 
    NamedTuple, 
    Literal, 
    TypedDict, 
    Generator, 
    Protocol, 
    TypeVar, 
    Generic,
    ContextManager,
    TypeAlias,
    TYPE_CHECKING
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]
from statictorch import (
    anify,
    TensorDimensionDescriptor,
    Tensor0d,
    Tensor1d,
    Tensor2d,
    Tensor3d
)

import torch
import csfile
import csdir
import io
import csv
import numpy
import librosa
import scipy.signal
import abc
import itertools
import sys
import time
import matplotlib.pyplot
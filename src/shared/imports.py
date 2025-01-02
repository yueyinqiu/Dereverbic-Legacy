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
    Generic
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW  # type: ignore

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
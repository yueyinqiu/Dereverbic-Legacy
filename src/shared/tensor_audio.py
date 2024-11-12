from pathlib import Path as _Path
from typing import Literal as _Literal
from torch import Tensor as _Tensor


def load_audio(path: _Path,
               sample_rate: int,
               mutichannel_behavior: _Literal["first_only", "as_mono", "as_many"]) -> _Tensor:
    """
    Always return a 2d tensor.
    """
    import librosa
    import numpy
    import torch

    numpy_: numpy.ndarray
    numpy_, _ = librosa.load(path, 
                             sr=sample_rate, 
                             mono=mutichannel_behavior == "as_mono")
    result: torch.Tensor = torch.tensor(data=numpy_, dtype=torch.float)

    if result.shape.__len__() == 1:
        result = result.unsqueeze(0)
    elif mutichannel_behavior == "first_only":
        result = result[0:1, :]
    
    return result


def save_audio(audio: _Tensor, path: _Path, sample_rate: int) -> None:
    import soundfile
    soundfile.write(path, audio.numpy(), sample_rate)

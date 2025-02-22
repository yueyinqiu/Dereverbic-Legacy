from pathlib import Path
from typing import Literal

import numpy
from statictorch import Tensor1d, Tensor2d
from torch import Tensor

from basic_utilities.static_class import StaticClass


class TensorAudios(StaticClass):
    @classmethod
    def load_audio(cls,
                   path: Path,
                   sample_rate: int,
                   mutichannel_behavior: Literal["first_only", "as_mono", "as_many"]) \
                    -> Tensor2d:
        import librosa
        import numpy
        import torch

        numpy_: numpy.ndarray
        numpy_, _ = librosa.load(path, 
                                 sr=sample_rate, 
                                 mono=mutichannel_behavior == "as_mono")
        result: Tensor = torch.tensor(data=numpy_, dtype=torch.float)

        if result.shape.__len__() == 1:
            result = result.unsqueeze(0)
        elif mutichannel_behavior == "first_only":
            result = result[0:1, :]
        
        return Tensor2d(result)

    @classmethod
    def save_audio(cls, 
                   audio: Tensor1d | Tensor2d, 
                   path: Path, 
                   sample_rate: int) -> None:
        audio_numpy: numpy.ndarray = audio.detach().cpu().numpy()
        if audio_numpy.shape.__len__() == 2 and audio_numpy.shape[0] == 1:
            audio_numpy = audio_numpy.squeeze(0)
        import soundfile
        soundfile.write(path, audio_numpy, sample_rate)

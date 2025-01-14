from .i0 import *
from .static_class import StaticClass

class TensorAudio(StaticClass):
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
        import soundfile
        soundfile.write(path, audio.detach().cpu().numpy(), sample_rate)

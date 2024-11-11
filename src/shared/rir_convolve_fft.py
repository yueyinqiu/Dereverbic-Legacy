import torch as _torch
from torch import Tensor as _Tensor
import numpy as _numpy
from typing import Literal as _Literal
from numpy import ndarray as _ndarray

def convolve(a: _Tensor, 
             v: _Tensor,
             mode: _Literal["same", "valid", "full"] = "same") -> _Tensor:
    # v -> rir
    # a -> speech

    n_fft: int = a.shape[-1] + v.shape[-1] - 1

    rir_fft: _Tensor = _torch.fft.fft(v, n=n_fft)
    speech_fft: _Tensor = _torch.fft.fft(a, n=n_fft)

    reverb_fft: _Tensor = rir_fft * speech_fft
    reverb_fft_ifft: _Tensor = _torch.fft.ifft(reverb_fft)

    center_index: int = n_fft // 2
    if mode == "same":
        length: int = max(a.shape[-1], v.shape[-1])
    elif mode == "valid":
        length = max(a.shape[-1], v.shape[-1]) - min(a.shape[-1], v.shape[-1]) + 1
    else:
        length = n_fft
    
    if n_fft % 2 == 1:
        left: int = center_index - length // 2
        right: int = center_index + length // 2 + length % 2
    else:
        left = center_index - length // 2 - length % 2
        right = center_index + length // 2
    return reverb_fft_ifft[..., left:right].real

def inverse_convolve_full(a_star_v: _Tensor,
                          a_or_v: _Tensor):
    # a_star_v -> reverb
    # a_or_v -> speech

    n_fft: int = a_star_v.shape[-1]

    reverb_fft: _Tensor = _torch.fft.fft(a_star_v, n=n_fft)
    speech_fft: _Tensor = _torch.fft.fft(a_or_v, n=n_fft)

    rir_fft: _Tensor = reverb_fft / speech_fft
    rir_fft_ifft: _Tensor = _torch.fft.ifft(rir_fft)

    return rir_fft_ifft[0:a_star_v.shape[-1] - a_or_v.shape[-1] + 1].real


def get_reverb(speech: _Tensor, 
               rir: _Tensor,
               cut: bool = True):
    reverb: _Tensor = convolve(speech, rir, "full")
    if cut:
        reverb = reverb[..., :speech.shape[-1]]
    return reverb


def _test_convolve():
    import random
    random_: random.Random = random.Random(1234)

    speech_list: list[float] = [random_.random() for _ in range(1000)]
    rir_list: list[float] = [random_.random() for _ in range(1000)]

    def test_on(speech_length: int,
                rir_length: int, 
                mode: _Literal["same", "valid", "full"]):
        speech_numpy: _ndarray = _numpy.array(speech_list[:speech_length])
        rir_numpy: _ndarray = _numpy.array(rir_list[:rir_length])
        result_numpy: _ndarray = _numpy.convolve(speech_numpy, rir_numpy, mode)

        speech_torch: _Tensor = _torch.tensor(speech_numpy)
        rir_torch: _Tensor = _torch.tensor(rir_numpy)
        result_torch: _Tensor = convolve(speech_torch, rir_torch, mode)
        print(f"Speech: {speech_length}    Rir: {rir_length}    Mode: {mode}")

        difference: _ndarray = (result_torch.numpy() - result_numpy)
        argmax: _numpy.intp = difference.__abs__().argmax()
        print(f"Difference:", end=" ")
        print(f"{result_torch.numpy()[argmax]:.3e} - {result_numpy[argmax]:.3e} =", end=" ")
        print(f"{difference[argmax]:.3e}", end=" ")
        print(f"({(difference[argmax] / result_numpy[argmax] * 100):.5f}%)")

    print("====== _test_convolve ======")

    test_on(800, 200, "same")
    test_on(800, 200, "valid")
    test_on(800, 200, "full")
    print()

    test_on(800, 201, "same")
    test_on(800, 201, "valid")
    test_on(800, 201, "full")
    print()

    test_on(801, 200, "same")
    test_on(801, 200, "valid")
    test_on(801, 200, "full")
    print()


def _test_inverse_convolve():
    import random
    random_: random.Random = random.Random(1234)

    speech_list: list[float] = [random_.random() for _ in range(1000)]
    rir_list: list[float] = [random_.random() for _ in range(1000)]

    def test_on(speech_length: int,
                rir_length: int):
        speech: _Tensor = _torch.tensor(speech_list[:speech_length])
        rir: _Tensor = _torch.tensor(rir_list[:rir_length])
        reverb: _Tensor = convolve(speech, rir, "full")
        
        rir_: _Tensor = inverse_convolve_full(reverb, speech)
        rir_ = rir_[:rir_length]

        difference: _Tensor = rir_ - rir
        argmax: _Tensor = _torch.argmax(difference.abs())
        print(f"Speech: {speech_length}    Rir: {rir_length}")
        print(f"Difference:", end=" ")
        print(f"{rir_.numpy()[argmax]:.3e} - {rir[argmax]:.3e} =", end=" ")
        print(f"{difference[argmax]:.3e}", end=" ")
        print(f"({(difference[argmax] / rir[argmax] * 100):.5f}%)")

    print("====== _test_inverse_convolve ======")

    test_on(800, 200)
    test_on(800, 201)
    test_on(801, 200)
    print()


if __name__ == "__main__":
    _test_convolve()
    _test_inverse_convolve()

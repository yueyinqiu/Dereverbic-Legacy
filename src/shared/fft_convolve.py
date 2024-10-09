import torch
import torch.nn.functional
import numpy
import typing

def convolve(speech: torch.Tensor, 
             rir: torch.Tensor,
             mode: typing.Literal["same", "valid", "full"] = "same") -> torch.Tensor:
    n_fft: int = speech.shape[-1] + rir.shape[-1] - 1

    rir_fft: torch.Tensor = torch.fft.fft(rir, n=n_fft)
    speech_fft: torch.Tensor = torch.fft.fft(speech, n=n_fft)

    reverb_fft: torch.Tensor = rir_fft * speech_fft
    reverb_fft_ifft: torch.Tensor = torch.fft.ifft(reverb_fft)

    center_index: int = n_fft // 2
    if mode == "same":
        length: int = max(speech.shape[-1], rir.shape[-1])
    elif mode == "valid":
        length = max(speech.shape[-1], rir.shape[-1]) - min(speech.shape[-1], rir.shape[-1]) + 1
    else:
        length = speech.shape[-1] + rir.shape[-1] - 1
    
    if n_fft % 2 == 1:
        left: int = center_index - length // 2
        right: int = center_index + length // 2 + length % 2
    else:
        left = center_index - length // 2 - length % 2
        right = center_index + length // 2
    return reverb_fft_ifft[left:right].real

def inverse_convolve_full(reverb: torch.Tensor,
                          speech: torch.Tensor,
                          ir_length: int):
    n_fft: int = speech.shape[-1] + ir_length - 1

    reverb_fft: torch.Tensor = torch.fft.fft(reverb, n=n_fft)
    speech_fft: torch.Tensor = torch.fft.fft(speech, n=n_fft)

    rir_fft: torch.Tensor = reverb_fft / speech_fft
    rir_fft_ifft: torch.Tensor = torch.fft.ifft(rir_fft)

    return rir_fft_ifft[0:ir_length].real


def _test_convolve():
    import random
    random_: random.Random = random.Random(1234)

    speech_list: list[float] = [random_.random() for _ in range(1000)]
    rir_list: list[float] = [random_.random() for _ in range(1000)]

    def test_on(speech_length: int,
                rir_length: int, 
                mode: typing.Literal["same", "valid", "full"]):
        speech_numpy: numpy.ndarray = numpy.array(speech_list[:speech_length])
        rir_numpy: numpy.ndarray = numpy.array(rir_list[:rir_length])
        result_numpy: numpy.ndarray = numpy.convolve(speech_numpy, rir_numpy, mode)

        speech_torch: torch.Tensor = torch.tensor(speech_numpy)
        rir_torch: torch.Tensor = torch.tensor(rir_numpy)
        result_torch: torch.Tensor = convolve(speech_torch, rir_torch, mode)
        print(f"Speech: {speech_length}    Rir: {rir_length}    Mode: {mode}")

        difference: numpy.ndarray = (result_torch.numpy() - result_numpy)
        argmax: numpy.intp = difference.__abs__().argmax()
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
        speech: torch.Tensor = torch.tensor(speech_list[:speech_length])
        rir: torch.Tensor = torch.tensor(rir_list[:rir_length])
        reverb: torch.Tensor = convolve(speech, rir, "full")
        
        rir_: torch.Tensor = inverse_convolve_full(reverb, speech, rir_length)
        rir_ = rir_[:rir_length]

        difference: torch.Tensor = rir_ - rir
        argmax: torch.Tensor = torch.argmax(difference.abs())
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

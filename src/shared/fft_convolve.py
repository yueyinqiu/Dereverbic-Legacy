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

    conv_fft: torch.Tensor = rir_fft * speech_fft
    conv_fft_ifft: torch.Tensor = torch.fft.ifft(conv_fft)

    center_index: int = n_fft // 2
    if mode == "same":
        conv_length: int = max(speech.shape[-1], rir.shape[-1])
    elif mode == "valid":
        conv_length = max(speech.shape[-1], rir.shape[-1]) - min(speech.shape[-1], rir.shape[-1]) + 1
    else:
        conv_length = speech.shape[-1] + rir.shape[-1] - 1
    
    if n_fft % 2 == 1:
        conv_fft_ifft = conv_fft_ifft[
            center_index - conv_length // 2 : center_index + conv_length // 2 + conv_length % 2]
    else:
        conv_fft_ifft = conv_fft_ifft[
            center_index - conv_length // 2 - conv_length % 2: center_index + conv_length // 2]
    return conv_fft_ifft.real


def _test():
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
        print(f"{result_torch.numpy()[argmax]} - {result_numpy[argmax]} =", end=" ")
        print(f"{difference[argmax]}", end=" ")
        print(f"({difference[argmax] / result_numpy[argmax]})")

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


if __name__ == "__main__":
    _test()

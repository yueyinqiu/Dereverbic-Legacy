from .imports import *
from .static_class import StaticClass


class RirConvolveFft(StaticClass):
    @classmethod
    def convolve(cls,
                 a: Tensor, 
                 v: Tensor,
                 mode: Literal["same", "valid", "full"] = "same") -> Tensor:
        # v -> rir
        # a -> speech

        n_fft: int = a.shape[-1] + v.shape[-1] - 1

        rir_fft: Tensor = torch.fft.fft(v, n=n_fft)
        speech_fft: Tensor = torch.fft.fft(a, n=n_fft)

        reverb_fft: Tensor = rir_fft * speech_fft
        reverb_fft_ifft: Tensor = torch.fft.ifft(reverb_fft)

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

    @classmethod
    def inverse_convolve_full(cls,
                              a_star_v: Tensor,
                              a_or_v: Tensor):
        # a_star_v -> reverb
        # a_or_v -> speech

        n_fft: int = a_star_v.shape[-1]

        reverb_fft: Tensor = torch.fft.fft(a_star_v, n=n_fft)
        speech_fft: Tensor = torch.fft.fft(a_or_v, n=n_fft)

        rir_fft: Tensor = reverb_fft / speech_fft
        rir_fft_ifft: Tensor = torch.fft.ifft(rir_fft)

        return rir_fft_ifft[0:a_star_v.shape[-1] - a_or_v.shape[-1] + 1].real

    @classmethod
    def get_reverb(cls,
                   speech: Tensor, 
                   rir: Tensor,
                   cut: bool = True):
        reverb: Tensor = cls.convolve(speech, rir, "full")
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
                mode: Literal["same", "valid", "full"]):
        speech_numpy: numpy.ndarray = numpy.array(speech_list[:speech_length])
        rir_numpy: numpy.ndarray = numpy.array(rir_list[:rir_length])
        result_numpy: numpy.ndarray = numpy.convolve(speech_numpy, rir_numpy, mode)

        speech_torch: Tensor = torch.tensor(speech_numpy)
        rir_torch: Tensor = torch.tensor(rir_numpy)
        result_torch: Tensor = RirConvolveFft.convolve(speech_torch, rir_torch, mode)
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
        speech: Tensor = torch.tensor(speech_list[:speech_length])
        rir: Tensor = torch.tensor(rir_list[:rir_length])
        reverb: Tensor = RirConvolveFft.convolve(speech, rir, "full")
        
        rir_: Tensor = RirConvolveFft.inverse_convolve_full(reverb, speech)
        rir_ = rir_[:rir_length]

        difference: Tensor = rir_ - rir
        argmax: Tensor = torch.argmax(difference.abs())
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

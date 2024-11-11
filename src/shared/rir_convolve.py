from torch import Tensor as _Tensor


def get_reverb(speech: _Tensor, 
               rir: _Tensor,
               cut: bool = True):
    import torch

    assert speech.shape.__len__() == 1
    assert rir.shape.__len__() == 1

    speech = speech.unsqueeze(0).unsqueeze_(0)
    rir = rir.flip(0).unsqueeze_(0).unsqueeze_(0)

    reverb: _Tensor = torch.conv1d(speech, rir, padding=(rir.shape[-1] - 1))
    reverb.squeeze_()
    if cut:
        reverb = reverb[:speech.shape[-1]]
    return reverb


def _test():
    import torch
    import numpy

    speech: numpy.ndarray = numpy.array([1, 2, 3, 4, 5])
    rir: numpy.ndarray = numpy.array([3, 2, 1])
    expected: numpy.ndarray = numpy.convolve(speech, rir, "full")
    
    actual: _Tensor = get_reverb(torch.tensor(speech),
                                 torch.tensor(rir),
                                 False)
    print((expected - actual.numpy()).max())
    print((expected - actual.numpy()).argmax())

if __name__ == "__main__":
    _test()

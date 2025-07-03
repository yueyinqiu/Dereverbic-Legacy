import numpy
import scipy.signal
import torch


class BerpButterWorthFilter(torch.nn.Module):
    def __init__(self, fs: int = 16000, fc: int = 128):
        self.fc = fc
        self.fs = fs

    def lpf(self):
        if self.fc <= 200:
            N: int = 6
        elif self.fc > 200 and self.fc <= 700:
            N = 9
        else:
            N = 12
        Bd: numpy.ndarray
        Ad: numpy.ndarray
        Bd, Ad = scipy.signal.butter(N,  # pyright: ignore [reportGeneralTypeIssues, reportAssignmentType]
                                     Wn=self.fc, 
                                     btype="lowpass", 
                                     fs=self.fs, 
                                     output="ba")
        return torch.as_tensor(Bd), torch.as_tensor(Ad)
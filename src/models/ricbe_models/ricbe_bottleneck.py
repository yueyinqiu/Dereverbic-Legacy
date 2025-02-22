# Inspired by https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from statictorch import Tensor3d
import torch


class RicbeBottleneck(torch.nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()        
        kernel_size: int = 3
        self.conv = torch.nn.Conv1d(channels, 
                                    channels, 
                                    kernel_size=kernel_size, 
                                    padding=(((kernel_size - 1) // 2) * dilation), 
                                    dilation=dilation)
        self.prelu = torch.nn.PReLU(channels)
        self.lstm = torch.nn.LSTM(channels, channels, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(channels, channels)
    
    def forward(self, x: Tensor3d):
        x = self.conv(x)
        x = self.prelu(x)
        x = Tensor3d(x.permute(0, 2, 1))
        x, _ = self.lstm(x)
        x = self.linear(x)
        return Tensor3d(x.permute(0, 2, 1))

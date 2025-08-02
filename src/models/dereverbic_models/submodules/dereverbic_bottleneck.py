from statictorch import Tensor3d
import torch

from models.dereverbic_models.submodules.dereverbic_decoder import DereverbicDecoder
from models.dereverbic_models.submodules.dereverbic_encoder import DereverbicEncoder


class DereverbicBottleneck(torch.nn.Module):
    def __init__(self, channels: int, dilation: int, replace_lstm_with_encoder_decoder: bool) -> None:
        super().__init__()        
        kernel_size: int = 3
        self.conv = torch.nn.Conv1d(channels, 
                                    channels, 
                                    kernel_size=kernel_size, 
                                    padding=(((kernel_size - 1) // 2) * dilation), 
                                    dilation=dilation)
        self.prelu = torch.nn.PReLU(channels)

        self.use_lstm: bool
        if not replace_lstm_with_encoder_decoder:
            self.use_lstm = True
            self.lstm = torch.nn.LSTM(channels, channels, num_layers=2, batch_first=True)
            self.linear = torch.nn.Linear(channels, channels)
        else:
            self.use_lstm = False
            block_count: int = 3
            channel_step: int = 48
            self.encoder = DereverbicEncoder(block_count, channels, channel_step, 1)
            final_channels: int = block_count * channel_step + channels
            self.decoder = DereverbicDecoder(block_count, final_channels, channel_step, 1, True, False)
            self.lstm = torch.nn.Sequential()
    
    def forward(self, x: Tensor3d):
        x = self.conv(x)
        x = self.prelu(x)
        if self.use_lstm:
            x = Tensor3d(x.permute(0, 2, 1))
            x, _ = self.lstm(x)
            x = self.linear(x)
            return Tensor3d(x.permute(0, 2, 1))
        else:
            features: list[Tensor3d] = self.encoder(x)
            x = self.decoder(features[-1], features)
            return Tensor3d(x)

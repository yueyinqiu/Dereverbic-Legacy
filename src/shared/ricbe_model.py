# Modified from https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from .i0 import *
from .rir_blind_estimation_model import RirBlindEstimationModel
from .mrstft_loss import MrstftLoss


class AutoVerb(torch.nn.Module):
    def __init__(self, blocks: int, in_channels: int, channel_factor: int):
        super().__init__()
        dilation: int = 1
        self.preprocess = torch.nn.Sequential(
            torch.nn.Conv1d(1, in_channels, kernel_size=5, padding=2, stride=1),
            torch.nn.PReLU(in_channels)
        )
        self.encoder = Encoder(blocks, in_channels, channel_factor, dilation)
        self.bottlenect = Bottleneck(in_channels, dilation)
        self.decoder = Decoder(blocks, self.encoder.get_channels_output(), channel_factor, dilation)
        self.postprocess = CutBlock(in_channels, 1)

    def forward(self, mix: Tensor3d):
        # [32, 48, 80000]
        x: Tensor3d = self.preprocess(mix)
        features: list[Tensor3d] = self.encoder(x)

        # [32, 288, 79]
        x = self.bottlenect(features[-1])

        x = self.decoder(x, features)
        
        x = self.postprocess(x)
        return x

class EncoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dilation: int):
        super().__init__()
        kernel: int = 7
        stride: int = 4
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels, 
                                     kernel_size=kernel, 
                                     stride=stride,
                                     dilation=dilation,
                                     padding=((kernel - 1) // 2) * dilation)
        self.prelu1 = torch.nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu1(self.conv1(x))

class Encoder(torch.nn.Module):
    class EncoderBlockList(Protocol):
        def __iter__(self) -> Iterator[EncoderBlock]:
            raise RuntimeError()

    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_increase_per_layer: int, 
                 dilation: int) -> None:
        super().__init__()
        block_list: list[EncoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input + channels_increase_per_layer
            block_list.append(EncoderBlock(channels_input, channels_next, dilation))
            channels_input = channels_next
        self._channels_output = channels_input
        self.blocks: Encoder.EncoderBlockList = anify(torch.nn.ModuleList(block_list))

    def get_channels_output(self):
        return self._channels_output

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = []
        features.append(x)
        block: EncoderBlock
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class Bottleneck(torch.nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()        
        kernel: int = 3
        self.conv = torch.nn.Conv1d(channels, 
                                    channels, 
                                    kernel_size=kernel, 
                                    padding=(((kernel - 1) // 2) * dilation), 
                                    dilation=dilation)
        self.prelu = torch.nn.PReLU(channels)
        self.lstm = torch.nn.LSTM(channels, channels, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(channels, channels)
    
    def forward(self, x: Tensor3d):
        x = self.conv(x)
        # [32, 288, 79]
        x = self.prelu(x)
        # [32, 79, 288]
        x = Tensor3d(x.permute(0, 2, 1))
        # [32, 79, 288]
        x, _ = self.lstm(x)
        # [32, 79, 288]
        x = self.linear(x)
        # [32, 288, 79]
        return Tensor3d(x.permute(0, 2, 1))

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        kernel: int = 7
        stride: int = 1
        padding: int = ((kernel - 1) // 2) * dilation

        self.upsample = torch.nn.Upsample(scale_factor = 4)
        self.conv1 = torch.nn.Conv1d(in_channels, 
                                     out_channels, 
                                     kernel_size=kernel, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)
        self.conv2 = torch.nn.Conv1d(out_channels, 
                                     out_channels, 
                                     kernel_size=kernel, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)
        self.conv3 = torch.nn.Conv1d(out_channels, 
                                     out_channels,
                                     kernel_size=kernel, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)

        self.prelu1 = torch.nn.PReLU(out_channels)
        self.prelu2 = torch.nn.PReLU(out_channels)
        self.prelu3 = torch.nn.PReLU(out_channels)

    def forward(self, x: Tensor3d):
        y1: Tensor3d = self.upsample(x)
        y1 = self.prelu1(self.conv1(y1))
        y2: Tensor3d = self.prelu2((self.conv2(y1)))
        y2 = self.prelu3(self.conv3(y2))
        return y1 + y2

class Decoder(torch.nn.Module):
    class DecoderBlockList(Protocol):
        def __iter__(self) -> Iterator[DecoderBlock]:
            raise RuntimeError()
        
    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_decrease_per_layer: int, 
                 dilation: int):
        super().__init__()
        block_list: list[DecoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input - channels_decrease_per_layer
            block_list.append(DecoderBlock(channels_input, channels_next, dilation))
            channels_input = channels_next
        self.blocks: Decoder.DecoderBlockList = anify(torch.nn.ModuleList(block_list))

    def forward(self, x: Tensor3d, features: list[Tensor3d]):
        i: int = -1
        block: DecoderBlock
        for block in self.blocks:
            x = Tensor3d(x + features[-i])
            x = Tensor3d(block(x))
            i = i - 1
            x = Tensor3d(x[:, :, :features[-i].size(2)])
        return Tensor3d(x + features[-i])

class CutBlock(torch.nn.Module):
    def __init__(self, channels, out, dil=1, stride=1, mul=1):
        super().__init__()
        self.first_kernel = 11
        self.second_kernel = 3
        self.conv1 = torch.nn.Conv1d(channels, channels // 2, kernel_size = self.first_kernel, stride = stride, padding = ((self.first_kernel - 1) // 2) * dil)
        self.conv2 = torch.nn.Conv1d(channels // 2, out, kernel_size = self.second_kernel, stride = stride, padding = ((self.second_kernel - 1) // 2) * dil)
        self.prelu = torch.nn.PReLU(channels // 2)
        
    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.conv2(x)
        return x

class RicbeModel(RirBlindEstimationModel):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = AutoVerb(blocks=5, in_channels=48, channel_factor=48).to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)
        self.random = torch.Generator(device).manual_seed(seed)
        self.spec_loss = MrstftLoss(device, 
                                    fft_sizes=[512, 1024, 2048, 4096], 
                                    hop_sizes=[50, 120, 240, 480], 
                                    win_lengths=[512, 1024, 2048, 4096])
        self.l1 = torch.nn.L1Loss().to(device)

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]
        random: Tensor

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "random": self.random.get_state()
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.random.set_state(state["random"])

    def _predict(self,
                 reverb_batch: Tensor2d):
        speech: Tensor3d = self.module(reverb_batch.unsqueeze(1))
        return Tensor2d(speech.squeeze(1)[:, :16000])

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch)
        loss_l1: Tensor0d = self.l1(predicted, rir_batch)
        loss_stft: Tensor0d
        loss_stft = self.spec_loss(predicted, rir_batch)["mag_loss"]
        loss_total: Tensor0d = Tensor0d(loss_l1 + loss_stft)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        result: dict[str, float] = {
            "loss_total": float(loss_total),
            "loss_l1": float(loss_l1),
            "loss_stft": float(loss_stft)
        }

        return result

    def evaluate_on(self, reverb_batch: Tensor2d):
        self.module.eval()
        predicted: Tensor2d = self._predict(reverb_batch)
        self.module.train()
        return predicted
    
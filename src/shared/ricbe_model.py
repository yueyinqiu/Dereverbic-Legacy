# Inspired by https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from .i0 import *
from .rir_blind_estimation_model import RirBlindEstimationModel
from .mrstft_loss import MrstftLoss


class _Preprocess(torch.nn.Module):
    def __init__(self, 
                 channels_input: int, 
                 channels_output: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels_input, 
                                    channels_output, 
                                    kernel_size=5, 
                                    padding=2, 
                                    stride=1)
        self.prelu = torch.nn.PReLU(channels_output)
        
    def forward(self, x: Tensor3d):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _EncoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dilation: int):
        super().__init__()
        kernel_size: int = 7
        stride: int = 4
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     dilation=dilation,
                                     padding=((kernel_size - 1) // 2) * dilation)
        self.prelu1 = torch.nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu1(self.conv1(x))


class _Encoder(torch.nn.Module):
    class EncoderBlockList(Protocol):
        def __iter__(self) -> Iterator[_EncoderBlock]:
            raise RuntimeError()

    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_increase_per_layer: int, 
                 dilation: int) -> None:
        super().__init__()
        block_list: list[_EncoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input + channels_increase_per_layer
            block_list.append(_EncoderBlock(channels_input, channels_next, dilation))
            channels_input = channels_next
        self.blocks: _Encoder.EncoderBlockList = anify(torch.nn.ModuleList(block_list))

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = []
        features.append(x)
        block: _EncoderBlock
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class _Bottleneck(torch.nn.Module):
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


class _DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        kernel_size: int = 7
        stride: int = 1
        padding: int = ((kernel_size - 1) // 2) * dilation

        self.upsample = torch.nn.Upsample(scale_factor = 4)
        self.conv1 = torch.nn.Conv1d(in_channels, 
                                     out_channels, 
                                     kernel_size=kernel_size, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)
        self.conv2 = torch.nn.Conv1d(out_channels, 
                                     out_channels, 
                                     kernel_size=kernel_size, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)
        self.conv3 = torch.nn.Conv1d(out_channels, 
                                     out_channels,
                                     kernel_size=kernel_size, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)

        self.prelu1 = torch.nn.PReLU(out_channels)
        self.prelu2 = torch.nn.PReLU(out_channels)
        self.prelu3 = torch.nn.PReLU(out_channels)

    def forward(self, x: Tensor3d):
        y1: Tensor3d = self.upsample(x)
        y1 = self.conv1(y1)
        y1 = self.prelu1(y1)
        y2: Tensor3d = self.conv2(y1)
        y2 = self.prelu2(y2)
        y2 = self.prelu3(self.conv3(y2))
        return y1 + y2


class _Decoder(torch.nn.Module):
    class DecoderBlockList(Protocol):
        def __iter__(self) -> Iterator[_DecoderBlock]:
            raise RuntimeError()
        
    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_decrease_per_layer: int, 
                 dilation: int):
        super().__init__()
        block_list: list[_DecoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input - channels_decrease_per_layer
            block_list.append(_DecoderBlock(channels_input * 2, channels_next, dilation))
            channels_input = channels_next
        self.blocks: _Decoder.DecoderBlockList = anify(torch.nn.ModuleList(block_list))

    def forward(self, x: Tensor3d, features: list[Tensor3d]):
        i: int = -1
        block: _DecoderBlock
        for block in self.blocks:
            x = Tensor3d(torch.cat([x, features[i]], dim=1))
            x = Tensor3d(block(x))
            i = i - 1
            x = Tensor3d(x[:, :, :features[i].size(2)])
        return Tensor3d(torch.cat([x, features[i]], dim=1))


class _Postprocess(torch.nn.Module):
    def __init__(self, 
                 channels_input: int, 
                 channels_output: int, 
                 dilation: int, 
                 stride1: int,
                 stride2: int):
        super().__init__()
        self.kernel_size_1 = 11
        self.kernel_size_2 = 3
        self.conv1 = torch.nn.Conv1d(channels_input, 
                                     channels_input // 4, 
                                     kernel_size=self.kernel_size_1, 
                                     stride=stride1, 
                                     padding=((self.kernel_size_1 - 1) // 2) * dilation)
        self.conv2 = torch.nn.Conv1d(channels_input // 4, 
                                     channels_output, 
                                     kernel_size=self.kernel_size_2, 
                                     stride=stride2, 
                                     padding=((self.kernel_size_2 - 1) // 2) * dilation)
        self.prelu = torch.nn.PReLU(channels_input // 4)
        
    def forward(self, x: Tensor3d):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        return x


class _EncoderDecoderPair(torch.nn.Module):
    def __init__(self, channels_input: int):
        super().__init__()

        block_count: int = 5
        channel_step: int = 48

        self.encoder = _Encoder(block_count, channels_input, channel_step, 1)
        bottleneck_channels: int = block_count * channel_step + channels_input

        self.bottleneck = _Bottleneck(bottleneck_channels, 1)
        self.decoder = _Decoder(block_count, bottleneck_channels, channel_step, 1)

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = self.encoder(x)
        x = self.bottleneck(features[-1])
        x = self.decoder(x, features)
        return x


class _RicbeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48
        self.preprocess_for_speech = _Preprocess(1, channels)
        self.pair_for_speech = _EncoderDecoderPair(channels)
        self.postprocess_for_speech = _Postprocess(channels * 2, 1, 1, 1, 1)
        
        self.preprocess_for_rir = _Preprocess(2, channels)
        self.pair_for_rir = _EncoderDecoderPair(channels)
        self.postprocess_for_rir = _Postprocess(channels * 2, 1, 1, 1, 5)


    def forward(self, reverb: Tensor3d):
        rev: Tensor3d = self.preprocess_for_speech(reverb)
        spe: Tensor3d = self.pair_for_speech(rev)
        spe = self.postprocess_for_speech(spe)

        rs: Tensor3d = Tensor3d(torch.cat([reverb, spe], dim=1))

        rs = self.preprocess_for_rir(rs)
        rir: Tensor3d = self.pair_for_rir(rs)
        rir = self.postprocess_for_rir(rir)
        return rir, spe
    

class RicbeModel(RirBlindEstimationModel):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = _RicbeModule().to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)

        self.speech_mrstft = MrstftLoss(device, 
                                        fft_sizes=[256, 512, 1024, 2048], 
                                        hop_sizes=[64, 128, 256, 512], 
                                        win_lengths=[256, 512, 1024, 2048],
                                        window="hann_window")
        self.rir_mrstft = MrstftLoss(device, 
                                     fft_sizes=[32, 256, 1024, 4096],
                                     hop_sizes=[16, 128, 512, 2048],
                                     win_lengths=[32, 256, 1024, 4096], 
                                     window="hann_window")
        self.l1 = torch.nn.L1Loss().to(device)

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    class Prediction(NamedTuple):
        rir: Tensor2d
        speech: Tensor2d

    def _predict(self,
                 reverb_batch: Tensor2d) -> Prediction:
        rir: Tensor3d
        speech: Tensor3d
        rir, speech = self.module(reverb_batch.unsqueeze(1))
        return RicbeModel.Prediction(Tensor2d(rir.squeeze(1)), 
                                     Tensor2d(speech.squeeze(1)))

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: RicbeModel.Prediction = self._predict(reverb_batch)

        rir_l1: Tensor0d = self.l1(predicted.rir, rir_batch)
        rir_mrstft: Tensor0d = self.rir_mrstft(predicted.rir, rir_batch).total()
        speech_l1: Tensor0d = self.l1(predicted.speech, speech_batch)
        speech_mrstft: Tensor0d = self.speech_mrstft(predicted.speech, speech_batch).total()
        
        loss_rir: Tensor0d = Tensor0d(rir_l1 + rir_mrstft)
        loss_speech: Tensor0d = Tensor0d(speech_l1 + speech_mrstft)
        loss_total: Tensor0d = Tensor0d(loss_rir + loss_speech)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return {
            "loss_total": float(loss_total),
            "loss_rir": float(loss_rir),
            "loss_speech": float(loss_speech)
        }

    def evaluate_on(self, reverb_batch: Tensor2d):
        self.module.eval()
        predicted: RicbeModel.Prediction = self._predict(reverb_batch)
        self.module.train()
        return predicted
    
from .i0 import *
from .rir_blind_estimation_model import RirBlindEstimationModel
from .mrstft_loss import MrstftLoss
from .rir_convolve_fft import RirConvolveFft


class RicbeEncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm=True):
        super(RicbeEncoderBlock, self).__init__()
        if use_batchnorm:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                torch.nn.BatchNorm1d(out_channels, track_running_stats=True),
                torch.nn.PReLU(),
            )
            self.skip_conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                torch.nn.BatchNorm1d(out_channels, track_running_stats=True),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                torch.nn.PReLU(),
            )
            self.skip_conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            )

    def forward(self, x: Tensor3d):
        """
        x: [32, C, L] (L = 80000, 40000, 20000, 10000, 5000, 2500, 1250, 625, 313, 157, 79, 40, 20)
        """
        # [32, C', L / 2]
        out: Tensor = self.conv(x)
        # [32, C', L / 2]
        skip_out: Tensor = self.skip_conv(x)
        # [32, C', L / 2]
        skip_out = out + skip_out
        return skip_out


class RicbeEncoder(torch.nn.Module):
    def __init__(self):
        super(RicbeEncoder, self).__init__()
        block_list: list[RicbeEncoderBlock] = []
        channels: list[int] = [1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512]

        i: int
        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm: bool = True
            else:
                use_batchnorm = False
            in_channels: int = channels[i]
            out_channels: int = channels[i + 1]
            curr_block: RicbeEncoderBlock = RicbeEncoderBlock(in_channels, out_channels, use_batchnorm)
            block_list.append(curr_block)

        self.encode = torch.nn.Sequential(*block_list)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512, 128)

    def forward(self, x: Tensor3d):
        """
        x: [32, 1, 80000]
        """
        # [32, 512, 10]
        out: Tensor = self.encode(x)
        # [32, 512, 1]
        out = self.pooling(out)
        # [32, 512]
        out = out.squeeze(-1)
        # [32, 128]
        out = self.fc(out)
        return out


class RicbeUpsampleNet(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, upsample_factor: int):
        super(RicbeUpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer: torch.nn.ConvTranspose1d = torch.nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2,
        )
        torch.nn.init.orthogonal_(layer.weight)
        self.layer = torch.nn.utils.spectral_norm(layer)

    def forward(self, inputs: Tensor):
        outputs: Tensor = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs


class RicbeConditionalBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features: int, condition_length: int):
        super().__init__()

        self.num_features = num_features
        self.condition_length = condition_length
        self.norm = torch.nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)

        self.layer = torch.nn.utils.spectral_norm(torch.nn.Linear(condition_length, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)
        self.layer.bias.data.zero_()

    def forward(self, inputs: Tensor, noise: Tensor):
        outputs: Tensor = self.norm(inputs)
        gamma: Tensor
        beta: Tensor
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)

        outputs = gamma * outputs + beta

        return outputs


class RicbeDecoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 upsample_factor: int, 
                 condition_length: int):
        super(RicbeDecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.condition_batchnorm1 = RicbeConditionalBatchNorm1d(in_channels, condition_length)

        self.first_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            RicbeUpsampleNet(in_channels, in_channels, upsample_factor),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.condition_batchnorm2 = RicbeConditionalBatchNorm1d(out_channels, condition_length)

        self.second_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = torch.nn.Sequential(
            RicbeUpsampleNet(in_channels, in_channels, upsample_factor),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.condition_batchnorm3 = RicbeConditionalBatchNorm1d(out_channels, condition_length)

        self.third_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.condition_batchnorm4 = RicbeConditionalBatchNorm1d(out_channels, condition_length)

        self.fourth_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=8, padding=56),
        )

    def forward(self, enc_out: Tensor, condition: Tensor):
        inputs: Tensor = enc_out

        outputs: Tensor = self.condition_batchnorm1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs: Tensor = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs


class RicbeDecoder(torch.nn.Module):
    def __init__(self, num_filters: int, cond_length: int, rir_length: int):
        super(RicbeDecoder, self).__init__()
        self.rir_length = rir_length

        self.preprocess = torch.nn.Conv1d(1, 512, kernel_size=15, padding=7)

        self.blocks = torch.nn.ModuleList(
            [
                # 134
                RicbeDecoderBlock(512, 512, 1, cond_length),
                # 134
                RicbeDecoderBlock(512, 512, 1, cond_length),
                # 134
                RicbeDecoderBlock(512, 256, 2, cond_length),
                # 268
                RicbeDecoderBlock(256, 256, 2, cond_length),
                # 536
                RicbeDecoderBlock(256, 256, 2, cond_length),
                # 1072
                RicbeDecoderBlock(256, 128, 3, cond_length),
                # 3216
                RicbeDecoderBlock(128, 64, 5, cond_length)
                # 16080
            ]
        )

        self.postprocess = torch.nn.Sequential(torch.nn.Conv1d(64, num_filters + 1, kernel_size=15, padding=7))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, v: Tensor, condition: Tensor):
        """
        v: [32, 1, 134]

        condition: [32, 144]
        """
        # [32, 512, 134]
        inputs: Tensor = self.preprocess(v)
        outputs: Tensor = inputs
        layer: torch.nn.Module
        for layer in self.blocks:
            # Final: [32, 64, 16080]
            outputs = layer(outputs, condition)
        # [32, 64, 16000]
        outputs = outputs[:, :, :self.rir_length]
        # [32, 11, 16000]
        outputs = self.postprocess(outputs)

        # [32, 1, 16000]
        direct_early: Tensor = outputs[:, 0:1]
        # [32, 10, 16000]
        late: Tensor = outputs[:, 1:]
        # [32, 10, 16000]
        late = self.sigmoid(late)

        return direct_early, late


class RicbeNetwork(torch.nn.Module):
    @staticmethod
    def _get_octave_filters():
        f_bounds: list[tuple[float, float]] = []
        f_bounds.append((22.3, 44.5))
        f_bounds.append((44.5, 88.4))
        f_bounds.append((88.4, 176.8))
        f_bounds.append((176.8, 353.6))
        f_bounds.append((353.6, 707.1))
        f_bounds.append((707.1, 1414.2))
        f_bounds.append((1414.2, 2828.4))
        f_bounds.append((2828.4, 5656.8))
        f_bounds.append((5656.8, 11313.6))
        f_bounds.append((11313.6, 22627.2))

        firs: list = []
        low: float
        high: float
        for low, high in f_bounds:
            fir: numpy.ndarray = scipy.signal.firwin(
                1023,
                numpy.array([low, high]),
                pass_zero="bandpass",  # pyright: ignore [reportArgumentType]
                window="hamming",
                fs=48000,
            )
            firs.append(fir)

        firs_np: numpy.ndarray = numpy.array(firs)
        firs_np = numpy.expand_dims(firs_np, 1)
        return firs_np

    def __init__(self):
        super(RicbeNetwork, self).__init__()

        rir_length: int = 16000
        early_length: int = 800
        decoder_input_length: int = 134
        num_filters: int = 10
        noise_condition_length: int = 16
        z_size: int = 128
        filter_order: int = 1023

        self.rir_length = rir_length
        self.noise_condition_length = noise_condition_length
        self.num_filters = num_filters

        # Learned decoder input
        self.decoder_input = torch.nn.Parameter(torch.randn((1, 1, decoder_input_length)))
        self.encoder = RicbeEncoder()

        self.decoder = RicbeDecoder(num_filters, noise_condition_length + z_size, rir_length)

        # Learned "octave-band" like filter
        self.filter = torch.nn.Conv1d(
            num_filters,
            num_filters,
            kernel_size=filter_order,
            stride=1,
            padding="same",
            groups=num_filters,
            bias=False,
        )
        # Octave band pass initialization
        self.filter.weight.data = torch.tensor(RicbeNetwork._get_octave_filters(), dtype=torch.float32)

        # Mask for direct and early part
        mask: Tensor3d = Tensor3d(torch.zeros((1, 1, rir_length)))
        mask[:, :, : early_length] = 1.0
        self.mask: Tensor3d
        self.register_buffer("mask", mask, False)

        self.output_conv = torch.nn.Conv1d(num_filters + 1, 1, kernel_size=1, stride=1)

    def forward(self, 
                reverb_batch: Tensor2d,
                stochastic_noise: Tensor3d,
                noise_condition: Tensor2d) \
                    -> tuple[Tensor2d, Tensor2d]:
        """
        reverb_batch: [32, 80000]

        stochastic_noise: [32, 10, 16000]

        noise_condition: [32, 16]
        """
        # [32, 1, 80000]
        reverb: Tensor = Tensor3d(reverb_batch.unsqueeze(1))

        # [32, 10, 16000]
        filtered_noise: Tensor = self.filter(stochastic_noise)
        # [32, 128]
        z: Tensor = self.encoder(reverb)
        # [32, 144 = 128 + 16]
        condition: Tensor = torch.cat([z, noise_condition], dim=-1)
        # [32, 1, 134]
        decoder_input: Tensor = self.decoder_input.repeat(reverb.size()[0], 1, 1)

        # [32, 1, 16000]
        direct_early: Tensor
        # [32, 10, 16000]
        late_mask: Tensor
        direct_early, late_mask = self.decoder(decoder_input, condition)
        # [32, 10, 16000]
        late_part: Tensor = filtered_noise * late_mask
        # [32, 1, 16000]
        direct_early = torch.mul(direct_early, self.mask)
        # [32, 11, 16000]
        rir: Tensor = torch.cat((direct_early, late_part), 1)

        # [32, 1, 16000]
        rir = self.output_conv(rir)

        # [32, 16000]
        rir = Tensor2d(rir.squeeze(1))
        speech: Tensor2d = RirConvolveFft.inverse_convolve(reverb_batch, rir)
        return rir, speech


class RicbeModel(RirBlindEstimationModel):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = RicbeNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters())
        self.random = torch.Generator(device).manual_seed(seed)
        self.loss = MrstftLoss(device)

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

    class Prediction(NamedTuple):
        rir: Tensor2d
        speech: Tensor2d

    def _predict(self,
                 reverb_batch: Tensor2d,
                 stochastic_noise_batch: Tensor3d | None = None, 
                 noise_condition: Tensor2d | None = None):
        b: int = reverb_batch.size()[0]

        if stochastic_noise_batch is None:
            stochastic_noise_batch = Tensor3d(
                torch.randn((b, 1, self.module.rir_length), 
                            generator=self.random,
                            device=self.device))
            stochastic_noise_batch = Tensor3d(
                stochastic_noise_batch.repeat(1, self.module.num_filters, 1))

        if noise_condition is None:
            noise_condition = Tensor2d(
                torch.randn((b, self.module.noise_condition_length), 
                            generator=self.random, 
                            device=self.device))
        rir: Tensor2d
        speech: Tensor2d
        rir, speech = self.module(reverb_batch)
        return RicbeModel.Prediction(rir, speech)

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: RicbeModel.Prediction = self._predict(reverb_batch)
        loss_rir: Tensor0d = self.loss(predicted.rir, rir_batch)["total"]
        loss_speech: Tensor0d = self.loss(predicted.speech, speech_batch)["total"]
        loss_total: Tensor0d = Tensor0d(loss_rir + loss_speech * 0.)

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5)
        self.optimizer.step()

        result: dict[str, float] = {
            "loss_total": float(loss_total),
            "loss_rir": float(loss_rir),
            "loss_speech": float(loss_speech),
        }

        return result

    def evaluate_on(self, reverb_batch: Tensor2d) -> Prediction:
        self.module.eval()
        predicted: RicbeModel.Prediction = self._predict(reverb_batch)
        self.module.train()
        return predicted
    
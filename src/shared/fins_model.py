# The model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license


from torch._tensor import Tensor
from .i0 import *
from .rir_blind_estimation_model import RirBlindEstimationModel
from .mrstft_loss import MrstftLoss
from .rir_convolve_fft import RirConvolveFft


class FinsEncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm=True):
        super(FinsEncoderBlock, self).__init__()
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

    def forward(self, x: Tensor):
        out: Tensor = self.conv(x)
        skip_out: Tensor = self.skip_conv(x)
        skip_out = out + skip_out
        return skip_out


class FinsEncoder(torch.nn.Module):
    def __init__(self):
        super(FinsEncoder, self).__init__()
        block_list: list[FinsEncoderBlock] = []
        channels: list[int] = [1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512]

        i: int
        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm: bool = True
            else:
                use_batchnorm = False
            in_channels: int = channels[i]
            out_channels: int = channels[i + 1]
            curr_block: FinsEncoderBlock = FinsEncoderBlock(in_channels, out_channels, use_batchnorm)
            block_list.append(curr_block)

        self.encode = torch.nn.Sequential(*block_list)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512, 128)

    def forward(self, x: Tensor):
        b: int = x.size()[0]
        out: Tensor = self.encode(x)
        out = self.pooling(out)
        out = out.view(b, -1)
        out = self.fc(out)
        return out


class FinsUpsampleNet(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, upsample_factor: int):
        super(FinsUpsampleNet, self).__init__()
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


class FinsConditionalBatchNorm1d(torch.nn.Module):
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


class FinsDecoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 upsample_factor: int, 
                 condition_length: int):
        super(FinsDecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.condition_batchnorm1 = FinsConditionalBatchNorm1d(in_channels, condition_length)

        self.first_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            FinsUpsampleNet(in_channels, in_channels, upsample_factor),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.condition_batchnorm2 = FinsConditionalBatchNorm1d(out_channels, condition_length)

        self.second_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = torch.nn.Sequential(
            FinsUpsampleNet(in_channels, in_channels, upsample_factor),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.condition_batchnorm3 = FinsConditionalBatchNorm1d(out_channels, condition_length)

        self.third_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.condition_batchnorm4 = FinsConditionalBatchNorm1d(out_channels, condition_length)

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


class FinsDecoder(torch.nn.Module):
    def __init__(self, num_filters: int, cond_length: int):
        super(FinsDecoder, self).__init__()

        self.preprocess = torch.nn.Conv1d(1, 512, kernel_size=15, padding=7)

        self.blocks = torch.nn.ModuleList(
            [
                FinsDecoderBlock(512, 512, 1, cond_length),
                FinsDecoderBlock(512, 512, 1, cond_length),

                FinsDecoderBlock(512, 256, 1, cond_length),

                FinsDecoderBlock(256, 256, 2, cond_length),
                FinsDecoderBlock(256, 256, 2, cond_length),

                FinsDecoderBlock(256, 128, 2, cond_length),

                FinsDecoderBlock(128, 64, 5, cond_length),
            ]
        )

        self.postprocess = torch.nn.Sequential(torch.nn.Conv1d(64, num_filters + 1, kernel_size=15, padding=7))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, v: Tensor, condition: Tensor):
        inputs: Tensor = self.preprocess(v)
        outputs: Tensor = inputs
        layer: torch.nn.Module
        for layer in self.blocks:
            outputs = layer(outputs, condition)
        outputs = self.postprocess(outputs)

        direct_early: Tensor = outputs[:, 0:1]
        late: Tensor = outputs[:, 1:]
        late = self.sigmoid(late)

        return direct_early, late


class FinsNetwork(torch.nn.Module):
    @staticmethod
    def __get_octave_filters():
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
        super(FinsNetwork, self).__init__()

        # If you want to modify the parameters here,
        # blocks in _Decoder should also be modified.
        rir_length: int = 16000
        early_length: int = 800
        decoder_input_length: int = 400
        num_filters: int = 10
        noise_condition_length: int = 16
        z_size: int = 128
        filter_order: int = 1023

        self.rir_length = rir_length
        self.noise_condition_length = noise_condition_length
        self.num_filters = num_filters

        # Learned decoder input
        self.decoder_input = torch.nn.Parameter(torch.randn((1, 1, decoder_input_length)))
        self.encoder = FinsEncoder()

        self.decoder = FinsDecoder(num_filters, noise_condition_length + z_size)

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
        self.filter.weight.data = torch.tensor(FinsNetwork.__get_octave_filters(), dtype=torch.float32)

        # Mask for direct and early part
        mask: Tensor3d = Tensor3d(torch.zeros((1, 1, rir_length)))
        mask[:, :, : early_length] = 1.0
        self.mask: Tensor3d
        self.register_buffer("mask", mask, False)

        self.output_conv = torch.nn.Conv1d(num_filters + 1, 1, kernel_size=1, stride=1)

    def forward(self, 
                x: Tensor3d, 
                stochastic_noise: Tensor3d, 
                noise_condition: Tensor2d) \
                    -> Tensor3d:
        # Filter random noise signal
        filtered_noise: Tensor = self.filter(stochastic_noise)

        # Encode the reverberated speech
        z: Tensor = self.encoder(x)

        # Make condition vector
        condition: Tensor = torch.cat([z, noise_condition], dim=-1)

        # Learnable decoder input. Repeat it in the batch dimension.
        decoder_input: Tensor = self.decoder_input.repeat(x.size()[0], 1, 1)

        # Generate RIR
        direct_early: Tensor
        late_mask: Tensor
        direct_early, late_mask = self.decoder(decoder_input, condition)

        # Apply mask to the filtered noise to get the late part
        late_part: Tensor = filtered_noise * late_mask

        # Zero out sample beyond 2400 for direct early part
        direct_early = torch.mul(direct_early, self.mask)
        # Concat direct,early with late and perform convolution
        rir: Tensor = torch.cat((direct_early, late_part), 1)

        # Sum
        rir = self.output_conv(rir)

        return Tensor3d(rir)


class FinsModel(RirBlindEstimationModel):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = FinsNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters(), lr=0.000055, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.8
        )
        self.random = torch.Generator(device).manual_seed(seed)
        self.loss = MrstftLoss(device)

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]
        scheduler: dict[str, Any]
        random: Tensor

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "random": self.random.get_state()
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.random.set_state(state["random"])

    def _predict(self, 
                 reverb_batch: Tensor2d, 
                 stochastic_noise_batch: Tensor3d | None, 
                 noise_condition: Tensor2d | None) \
                    -> Tensor2d:
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
        predicted: Tensor3d = self.module(
            reverb_batch.unsqueeze(1), 
            stochastic_noise_batch, 
            noise_condition)
        return Tensor2d(predicted.squeeze(1))

    @staticmethod
    def __add_noise(reverb_batch: Tensor2d, 
                    noise_seed: int) -> Tensor2d:
        noise_batch: Tensor
        snr_batch: Tensor
        def _():
            random: numpy.random.RandomState = numpy.random.RandomState(noise_seed)
            device: torch.device = reverb_batch.device
            noises: list[Tensor] = []
            snrs: list[float] = []
            for _ in range(0, reverb_batch.size(0)):
                if random.random() < 0.9:
                    min_snr: float = 0.0
                    max_snr: float = 30.0
                    beta: float = random.random() + 1.0
                    import colorednoise
                    noise_numpy: numpy.ndarray = colorednoise.powerlaw_psd_gaussian(beta, 
                                                                                    reverb_batch.size(-1), 
                                                                                    random_state=random)
                    noises.append(torch.tensor(noise_numpy, device=device, dtype=torch.float32))
                    snrs.append(random.random() * (max_snr - min_snr) + min_snr)
                else:
                    noises.append(torch.zeros([reverb_batch.size(-1)], device=device))
                    snrs.append(0)
            return torch.stack(noises), torch.tensor(snrs, device=device)
        noise_batch, snr_batch = _()

        mean_square_signal: Tensor = torch.mean(reverb_batch ** 2, dim=1)
        signal_level_db: Tensor = 10 * torch.log10(mean_square_signal)
        noise_db: Tensor = signal_level_db - snr_batch
        mean_square_noise: Tensor = torch.sqrt(10 ** (noise_db / 10))
        mean_square_noise = torch.unsqueeze(mean_square_noise, dim=1)
        mean_square_noise = mean_square_noise.repeat(1, reverb_batch.size(1))
        modified_noise: Tensor = torch.mul(noise_batch, mean_square_noise)

        return Tensor2d(reverb_batch + modified_noise)

    @staticmethod
    def preprocess(rir_batch: Tensor2d, 
                   speech_batch: Tensor2d, 
                   noise_seed: int | None) \
                    -> tuple[Tensor2d, 
                             Tensor2d, 
                             Tensor2d]:
        rir_batch = Tensor2d(rir_batch / (0.999 * rir_batch.abs().max(dim=1, keepdim=True).values))
        speech_batch = Tensor2d(speech_batch - speech_batch.mean(dim=1, keepdim=True))
        speech_batch = Tensor2d(speech_batch * 0.1)
        reverb_batch: Tensor2d = RirConvolveFft.get_reverb_batch(speech_batch, rir_batch)

        rms_level: float = 0.01
        reverb_batch = Tensor2d(
            torch.sqrt(
                reverb_batch.shape[1] * rms_level ** 2 / (torch.sum(reverb_batch ** 2, dim=1, keepdim=True) + 1e-7)
            ) * reverb_batch)
        
        if noise_seed is not None:
            reverb_batch = FinsModel.__add_noise(reverb_batch, noise_seed)

        return reverb_batch, rir_batch, speech_batch
        
    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        #noise_seed: int = int(torch.randint(0, 2147483647, [], 
        #                                    device=self.device, 
        #                                    generator=self.random))
        #reverb_batch, rir_batch, _ = FinsModel.preprocess(rir_batch, speech_batch, noise_seed)
        
        predicted: Tensor2d = self._predict(reverb_batch, None, None)
        losses: MrstftLoss.Return = self.loss(predicted, rir_batch)

        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5)
        self.optimizer.step()

        result: dict[str, float] = {
            "loss_total": float(losses["total"]),
            "loss_mag": float(losses["mag_loss"]),
            "loss_sc": float(losses["sc_loss"]),
            "lr": self.scheduler.get_last_lr()[0]
        }

        self.scheduler.step()
        return result

    def evaluate_on(self, reverb_batch: Tensor2d) -> Tensor2d:
        self.module.eval()
        predicted: Tensor2d = self._predict(reverb_batch, None, None)
        self.module.train()
        return predicted
    
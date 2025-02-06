from .i0 import *
from .rir_blind_estimation_model import RirBlindEstimationModel
from .mrstft_loss import MrstftLoss

class Resample1d(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 transpose: bool = False, 
                 padding: Literal["reflect", "valid"] = "reflect", 
                 trainable: bool = False):
        '''
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        '''
        super(Resample1d, self).__init__()

        assert kernel_size > 2
        assert kernel_size % 2 == 1

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff: float = 0.5 / stride

        filter: numpy.ndarray = self._build_sinc_filter(kernel_size, cutoff)
        filter = filter.reshape([1, 1, kernel_size]).repeat(channels, axis=0)
        self.filter = torch.nn.Parameter(torch.from_numpy(filter), requires_grad=trainable)
    
    def _build_sinc_filter(self, kernel_size: int, cutoff: float):
        # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
        # Sinc lowpass filter
        # Build sinc kernel
        assert kernel_size % 2 == 1

        m: int = kernel_size - 1
        filter: numpy.ndarray = numpy.zeros(kernel_size, dtype=numpy.float32)
        i: int
        for i in range(kernel_size):
            if i == m // 2:
                filter[i] = 2 * numpy.pi * cutoff
            else:
                filter[i] = (numpy.sin(2 * numpy.pi * cutoff * (i - m // 2)) / (i - m // 2)) * \
                        (0.42 - 0.5 * numpy.cos((2 * numpy.pi * i) / m) + 0.08 * numpy.cos(4 * numpy.pi * m))

        filter = filter / numpy.sum(filter)
        return filter

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad: int = (self.kernel_size - 1) // 2
            out = torch.nn.functional.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = torch.nn.functional.conv_transpose1d(out, 
                                                       self.filter, 
                                                       stride=self.stride, 
                                                       padding=0, 
                                                       groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:, :, diff_steps // 2:-diff_steps // 2]
        else:
            assert(input_size % self.stride == 1)
            out = torch.nn.functional.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        '''
        Returns the output dimensionality (number of timesteps) for a given input size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''
        assert(input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return ((input_size - 1) * self.stride + 1)
        else:
            assert(input_size % self.stride == 1) # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        '''
        Returns the input dimensionality (number of timesteps) for a given output size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''

        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

class ConvLayer(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.filter = torch.nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size-1)
        else:
            self.filter = torch.nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert(n_outputs % NORM_CHANNELS == 0)
            self.norm = torch.nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = torch.nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            out = torch.nn.functional.relu(self.norm((self.filter(x))))
        else: # Add your own variations here with elifs conditioned on "conv_type" parameter!
            assert(self.conv_type == "normal")
            out = torch.nn.functional.leaky_relu(self.filter(x))
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert(input_size > 1)
            curr_size = (input_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size

        # Conv
        curr_size = curr_size - self.kernel_size + 1 # o = i + p - k + 1
        assert (curr_size > 0)

        # Strided conv/decimation
        if not self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size

class UpsamplingBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = torch.nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = torch.nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def centre_crop(self, x, target):
        '''
        Center-crop 3-dim. input tensor along last axis so it fits the target tensor shape
        :param x: Input tensor
        :param target: Shape of this tensor will be used as target shape
        :return: Cropped input tensor
        '''
        if target is None:
            return x

        target_shape = target.shape
        diff = x.shape[-1] - target_shape[-1]
        assert (diff % 2 == 0)
        crop = diff // 2

        if crop == 0:
            return x
        if crop < 0:
            raise ArithmeticError

        return x[:, :, crop:-crop].contiguous()

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = self.centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, self.centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = torch.nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = torch.nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size, conv_type, res, separate=False, depth=1, strides=2):
        super(Waveunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = torch.nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = torch.nn.Module()

            module.downsampling_blocks = torch.nn.ModuleList()
            module.upsampling_blocks = torch.nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

            module.bottlenecks = torch.nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = torch.nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            assert inst is not None
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict


class WaveUNetModel(RirBlindEstimationModel):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = Waveunet(1, 
                              [32 * 2 ** i for i in range(0, 6)], 
                              1, 
                              ["speech", "rir"], 
                              kernel_size=5,
                              target_output_size=80000, 
                              depth=1, 
                              strides=4,
                              conv_type="gn", 
                              res="fixed", 
                              separate=True).to(device)
        self.optimizer = AdamW(self.module.parameters(), 1e-3)
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

    def _predict(self,
                 reverb_batch: Tensor2d):
        prediected: dict[str, Tensor3d] = self.module(reverb_batch.unsqueeze(1))
        return Tensor2d(prediected["rir"].squeeze(1)), Tensor2d(prediected["speech"].squeeze(1))

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted_rir: Tensor2d
        predicted_speech: Tensor2d
        predicted_rir, predicted_speech = self._predict(reverb_batch)
        loss_rir: Tensor0d = self.loss(predicted_rir, rir_batch)["total"]
        loss_speech: Tensor0d = self.loss(predicted_speech, speech_batch)["total"]
        loss_total: Tensor0d = Tensor0d(loss_rir + loss_speech)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        result: dict[str, float] = {
            "loss_total": float(loss_total),
            "loss_rir": float(loss_rir),
            "loss_speech": float(loss_total)
        }

        return result

    def evaluate_on(self, reverb_batch: Tensor2d):
        self.module.eval()
        predicted: tuple[Tensor2d, Tensor2d] = self._predict(reverb_batch)
        self.module.train()
        return predicted
    
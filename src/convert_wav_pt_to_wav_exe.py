import convert_wav_pt_to_wav_config as config
import soundfile
import torch

for input, output in config.files:
    tensor: torch.Tensor = torch.load(input)
    soundfile.write(output, tensor.numpy(), config.sample_rate)

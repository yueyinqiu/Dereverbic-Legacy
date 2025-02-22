from pathlib import Path

from statictorch import Tensor1d
import torch

import convert_wav_pt_to_wav_config
from inputs_and_outputs.tensor_audios.tensor_audios import TensorAudios

input: Path
output: Path
for input, output in convert_wav_pt_to_wav_config.files:
    input = input.absolute()
    output = output.absolute()
    print(f"Dealing with {input} (-> {output} )...")
    tensor: Tensor1d = torch.load(input, weights_only=True)
    TensorAudios.save_audio(tensor, output, 16000)

print("Completed.")
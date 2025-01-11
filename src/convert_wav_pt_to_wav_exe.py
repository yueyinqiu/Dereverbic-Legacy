import convert_wav_pt_to_wav_config as config
from shared import *

input: Path
output: Path
for input, output in config.files:
    input = input.absolute()
    output = output.absolute()
    print(f"Dealing with {input} (-> {output} )...")
    tensor: torch.Tensor = torch.load(input, weights_only=True)
    TensorAudio.save_audio(tensor, output, 16000)

print("Completed.")
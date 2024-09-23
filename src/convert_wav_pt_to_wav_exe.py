import convert_wav_pt_to_wav_config as config
import soundfile
import torch

for input, output in config.files:
    input = input.absolute()
    output = output.absolute()
    print(f"Dealing with {input} (-> {output} )...")
    tensor: torch.Tensor = torch.load(input, weights_only=True)
    soundfile.write(output, tensor.numpy(), config.sample_rate)

print("Completed.")
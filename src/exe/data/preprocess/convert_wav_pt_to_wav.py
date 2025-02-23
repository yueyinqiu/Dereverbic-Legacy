from pathlib import Path

from statictorch import Tensor1d
import torch

from inputs_and_outputs.tensor_audios.tensor_audios import TensorAudios


def main():
    from exe.data.preprocess import convert_wav_pt_to_wav_config as config
    input: Path
    output: Path
    for input, output in config.files:
        input = input.absolute()
        output = output.absolute()
        print(f"Dealing with {input} (-> {output} )...")
        tensor: Tensor1d = torch.load(input, weights_only=True)
        TensorAudios.save_audio(tensor, output, 16000)

    print("Completed.")


if __name__ == "__main__":
    main()

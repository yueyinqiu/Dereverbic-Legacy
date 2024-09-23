import typing
import pathlib
import random
import torch
import utilities.string_random
import csv
import csdir

def load_audio(path: pathlib.Path,
               sample_rate: int,
               mutichannel_behavior: typing.Literal["first_only", "as_mono", "as_many"]):
    import librosa
    numpy, _ = librosa.load(path, 
                            sr=sample_rate, 
                            mono=mutichannel_behavior == "as_mono")
    result = torch.tensor(data=numpy, dtype=torch.float)

    if result.shape.__len__() == 1:
        result = result.unsqueeze(0)
    elif mutichannel_behavior == "first_only":
        result = result[0:1, :]
    
    return result

def save_audio(audio: torch.Tensor,
               sample_rate: int,
               file_name_without_suffix: str,
               directory: pathlib.Path):
    directory = directory / file_name_without_suffix[0] / file_name_without_suffix[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file = directory / f"{file_name_without_suffix}.wav.pt"
    torch.save(audio, tensor_file)

    audio_file = directory / f"{file_name_without_suffix}.wav"
    import soundfile
    soundfile.write(audio_file, audio.numpy(), sample_rate)

    return (tensor_file, audio_file)

def main():
    import convert_rir_to_tensor_config as config
    
    rand = random.Random(config.random_seed)
    string_random = utilities.string_random.StringRandom(rand, 16)

    with open(config.output_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as contents_file:
        contents_writer = csv.writer(contents_file)
        contents_writer.writerow([
            "Tensor", "Audio", "Original Audio", "Original Channel"])

        for path in config.inputs:
            path = path.absolute()
            print(f"Dealing with {path} ...")

            audio = load_audio(path, 
                               config.sample_rate, 
                               config.mutichannel_behavior)
            for i in range(audio.shape[0]):
                file_name = string_random.next()
                tensor_file, audio_file = save_audio(audio[i, :config.slice], 
                                                     config.sample_rate, 
                                                     file_name, 
                                                     config.output_directory)
                contents_writer.writerow([
                    str(tensor_file), str(audio_file), str(path), str(i)])
                
    print(f"Completed.")
    
main()
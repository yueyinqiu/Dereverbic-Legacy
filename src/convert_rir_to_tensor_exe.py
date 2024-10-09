import typing
import pathlib
import random
import torch
import shared.string_random
import csv
import csdir
import soundfile
import numpy
import _csv
import io

def load_audio(path: pathlib.Path,
               sample_rate: int,
               mutichannel_behavior: typing.Literal["first_only", "as_mono", "as_many"]):
    import librosa
    numpy_: numpy.ndarray
    numpy_, _ = librosa.load(path, 
                             sr=sample_rate, 
                             mono=mutichannel_behavior == "as_mono")
    result: torch.Tensor = torch.tensor(data=numpy_, dtype=torch.float)

    if result.shape.__len__() == 1:
        result = result.unsqueeze(0)
    elif mutichannel_behavior == "first_only":
        result = result[0:1, :]
    
    return result

def save_audio(audio: torch.Tensor,
               sample_rate: int,
               file_name_without_suffix: str,
               directory: pathlib.Path,
               save_wav: bool):
    directory = directory / file_name_without_suffix[0] / file_name_without_suffix[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file: pathlib.Path = directory / f"{file_name_without_suffix}.wav.pt"
    torch.save(audio, tensor_file)

    if save_wav:
        audio_file: str = str(directory / f"{file_name_without_suffix}.wav")
        soundfile.write(audio_file, audio.numpy(), sample_rate)
    else:
        audio_file = ""

    return (tensor_file, audio_file)

def main():
    import convert_rir_to_tensor_config as config
    
    rand: random.Random = random.Random(config.random_seed)
    string_random: shared.string_random.StringRandom = shared.string_random.StringRandom(rand, 16)

    csdir.create_directory(config.output_directory)
    contents_file: 'io.TextIOWrapper[io._WrappedBuffer]'
    with open(config.output_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as contents_file:
        contents_writer: '_csv._writer' = csv.writer(contents_file)
        contents_writer.writerow([
            "Tensor", "Audio", "Original Audio", "Original Channel"])

        path: pathlib.Path
        for path in config.inputs:
            path = path.absolute()
            print(f"Dealing with {path} ...")

            audio: torch.Tensor = load_audio(path, 
                                             config.sample_rate, 
                                             config.mutichannel_behavior)
            
            i: int
            for i in range(audio.shape[0]):
                file_name: str = string_random.next()
                tensor_file: pathlib.Path
                audio_file: str
                tensor_file, audio_file = save_audio(audio[i, :config.slice], 
                                                     config.sample_rate, 
                                                     file_name, 
                                                     config.output_directory,
                                                     config.save_wav)
                contents_writer.writerow([
                    str(tensor_file), str(audio_file), str(path), str(i)])
                
    print(f"Completed.")
    
main()
from shared.imports import *
import io
import csv
import _csv
import numpy
import librosa


def _save_tensor(audio: torch.Tensor,
                 file_name_without_suffix: str,
                 directory: Path):
    directory = directory / file_name_without_suffix[0] / file_name_without_suffix[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file: Path = directory / f"{file_name_without_suffix}.wav.pt"
    torch.save(audio, tensor_file)
    return tensor_file


def main():
    import convert_speech_to_tensor_config as config
    
    rand: Random = Random(config.random_seed)
    string_random: StringRandom = StringRandom(rand, 16)
    
    print("Sorting files ...")
    inputs: list[Path] = sorted(config.inputs)

    csdir.create_directory(config.output_directory)
    contents_file: io.TextIOWrapper
    with open(config.output_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as contents_file:
        contents_writer: '_csv._writer' = csv.writer(contents_file)
        contents_writer.writerow(["Tensor", "Original Audio"])

        path: Path
        for path in inputs:
            path = path.absolute()
            print(f"Dealing with {path} ...")

            audio: Tensor = tensor_audio.load_audio(path, 16000, "as_mono")
            channel: numpy.ndarray = audio[0, :].numpy()
            channel, _ = librosa.effects.trim(channel,
                                              top_db=60, 
                                              frame_length=2048, 
                                              hop_length=512)
            
            tensor: Tensor = torch.tensor(channel, dtype=torch.float)
            start: int = 16000 // 5
            while start + 5 * 16000 < tensor.__len__() - 16000 // 5:
                tensor_file: Path = _save_tensor(tensor[start:(start + 5 * 16000)], 
                                                 string_random.next(), 
                                                 config.output_directory)
                contents_writer.writerow([str(tensor_file), str(path)])
                contents_file.flush()
                start += 5 * 16000
                
    print(f"Completed.")
    
main()
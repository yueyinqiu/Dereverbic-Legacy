from shared.i import *
import _csv as _csv


def _save_tensor(audio: Tensor1d,
                 file_name_without_suffix: str,
                 directory: Path):
    directory = directory / file_name_without_suffix[0] / file_name_without_suffix[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file: Path = directory / f"{file_name_without_suffix}.wav.pt"
    torch.save(audio, tensor_file)
    return tensor_file


def main():
    import convert_rir_to_tensor_config as config
    
    rand: Random = Random(config.random_seed)
    string_random: StringRandom = StringRandom(rand, 16)

    print("Sorting files ...")
    inputs: list[Path] = sorted(config.inputs)

    csdir.create_directory(config.output_directory)
    contents_file: io.TextIOWrapper
    with open(config.output_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as contents_file:
        contents_writer: '_csv._writer' = csv.writer(contents_file)
        contents_writer.writerow(["Tensor", "Original Audio", "Original Channel"])

        path: Path
        for path in inputs:
            path = path.absolute()
            print(f"Dealing with {path} ...")

            audio: Tensor2d = TensorAudio.load_audio(path, 16000, "as_many")
            
            i: int
            for i in range(audio.shape[0]):
                tensor_file: Path = _save_tensor(Tensor1d(audio[i]), 
                                                 string_random.next(), 
                                                 config.output_directory)
                contents_writer.writerow([str(tensor_file), str(path), str(i)])
                contents_file.flush()
                
    print(f"Completed.")
    
main()
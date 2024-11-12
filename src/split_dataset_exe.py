from shared.imports import *
import csv
import _csv

def load_and_sort(contents: Path):
    tensor_files: list[str] = []

    csv_file: io.TextIOWrapper
    with open(contents, newline="") as csv_file:
        csv_reader: '_csv._reader' = csv.reader(csv_file)

        row_str: list[str]
        for row_str in csv_reader:
            assert tuple(row_str)[0] == "Tensor"
            assert tuple(row_str)[1] == "Original Audio"
            break
        
        for row_str in csv_reader:
            tensor_files.append(row_str[0])

    tensor_files.sort()
    return tensor_files

def save_reverb(rir_path: str, 
                speech_path: str,
                name_generator: StringRandom,
                directory: Path):
    rir: Tensor = torch.load(rir_path, weights_only=True)
    speech: Tensor = torch.load(speech_path, weights_only=True)
    reverb: Tensor = rir_convolve.get_reverb(speech, rir)

    file_name: str = name_generator.next()

    directory = directory / file_name[0] / file_name[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file: Path = directory / f"{file_name}.reverb.pt"
    torch.save({
        "rir": rir,
        "speech": speech,
        "reverb": reverb,
    }, tensor_file)
    return tensor_file


def main():
    import split_dataset_config as config
    
    random: Random = Random(config.random_seed)

    print("Shuffling...")
    # 文件名为随机值，排序就是打乱
    rirs: list[str] = load_and_sort(config.rir_contents)
    speeches: list[str] = load_and_sort(config.speech_contents)

    print("Saving train lists...")
    csfile.write_all_lines(config.train_list_rir, rirs[:-20000])
    csfile.write_all_lines(config.train_list_speech, speeches[:-20000])

    validation_files: list[str] = []
    test_files: list[str] = []
    reverb_name_generator: StringRandom = StringRandom(random, 16)
    reverb_contents_file: io.TextIOWrapper
    with open(config.reverb_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as reverb_contents_file:
        reverb_contents_writer: '_csv._writer' = csv.writer(reverb_contents_file)
        reverb_contents_writer.writerow(["Reverb", "Rir", "Speech"])

        print("Genrating validation files...")
        rir_path: str
        speech_path: str
        for rir_path, speech_path in zip(rirs[-20000:-10000], 
                                         speeches[-20000:-10000], 
                                         strict=True):
            print(f"Genrating for {rir_path} + {speech_path}...")
            reverb_path: Path = save_reverb(rir_path, 
                                            speech_path, 
                                            reverb_name_generator, 
                                            config.reverb_directory)
            reverb_contents_writer.writerow([str(reverb_path), rir_path, speech_path])
            reverb_contents_file.flush()
            validation_files.append(str(reverb_path))

        print("Genrating test files...")
        for rir_path, speech_path in zip(rirs[-10000:], 
                                         speeches[-10000:], 
                                         strict=True):
            print(f"Genrating for {rir_path} + {speech_path}...")
            reverb_path = save_reverb(rir_path, 
                                      speech_path, 
                                      reverb_name_generator, 
                                      config.reverb_directory)
            reverb_contents_writer.writerow([str(reverb_path), rir_path, speech_path])
            reverb_contents_file.flush()
            test_files.append(str(reverb_path))

    print("Saving validation lists...")
    csfile.write_all_lines(config.validation_list, validation_files)

    print("Saving test lists...")
    csfile.write_all_lines(config.test_list, test_files)


main()

from shared.i import *
import test_fins_config as config

print("# Loading...")
with torch.no_grad():
    random: Random = Random(config.random_seed)
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    model: FinsModel = FinsModel(config.device, random.randint(0, 1000))
    data: DataLoader = ValidationOrTestDataset(config.test_list, 
                                               config.device).get_data_loader(32)
    mrstft: MrstftLoss = MrstftLoss(config.device, 
                                    fft_sizes=[i * 16000 // 48000 for i in [64, 512, 2048, 8192]],
                                    hop_sizes=[i * 16000 // 48000 for i in [32, 256, 1024, 4096]],
                                    win_lengths=[i * 16000 // 48000 for i in [64, 512, 2048, 8192]],
                                    window="hann_window")
    print(f"# Batch count: {data.__len__()}")

    rank_file: Path = checkpoints.get_path(None) / "validation_rank.txt"
    if rank_file.exists():
        epoch: int = int(csfile.read_all_lines(rank_file)[0])
        path: Path = checkpoints.get_path(epoch)
        print(f"# Rank file found. The best checkpoint {epoch} will be used.")
    else:
        latest: CheckpointsDirectory.EpochAndPath | None = checkpoints.get_latest()
        if not latest:
            raise FileNotFoundError("Failed to find any checkpoint in the checkpoints directory.")
        epoch, path = latest
        print(f"# Failed to find the rank file. The latest checkpoint {epoch} will be used.")

    csv_print: CsvWriterProtocol = csv.writer(sys.stdout)
    csv_print.writerow(["batch", "metric", "value"])

    Trainer.load_model(model, path)

    mrstft_total_accumulator: KahanAccumulator = KahanAccumulator()
    mrstft_sc_accumulator: KahanAccumulator = KahanAccumulator()
    mrstft_mag_accumulator: KahanAccumulator = KahanAccumulator()

    batch_index: int
    batch: DataBatch
    for batch_index, batch in enumerate(data):
        predicted: Tensor2d = model.evaluate_on(batch.reverb)
        mrstft_value: MrstftLoss.Return = mrstft(predicted, batch.rir)
        
        mrstft_total: float = float(mrstft_value.total())
        mrstft_sc: float = float(mrstft_value.sc_loss)
        mrstft_mag: float = float(mrstft_value.mag_loss)

        mrstft_total_accumulator.add(mrstft_total)
        mrstft_sc_accumulator.add(mrstft_sc)
        mrstft_mag_accumulator.add(mrstft_mag)

        csv_print.writerow([batch_index, "mrstft_total", mrstft_total])
        csv_print.writerow([batch_index, "mrstft_sc", mrstft_sc])
        csv_print.writerow([batch_index, "mrstft_mag", mrstft_mag])

    csv_print.writerow(["average", "mrstft_total", 
                        mrstft_total_accumulator.value() / data.__len__()])
    csv_print.writerow(["average", "mrstft_sc", 
                        mrstft_sc_accumulator.value() / data.__len__()])
    csv_print.writerow(["average", "mrstft_mag", 
                        mrstft_mag_accumulator.value() / data.__len__()])

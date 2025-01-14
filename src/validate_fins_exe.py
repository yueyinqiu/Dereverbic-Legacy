from shared.i import *
import validate_fins_config as config

print("# Loading...")
with torch.no_grad():
    random: Random = Random(config.random_seed)
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    model: FinsModel = FinsModel(config.device, random.randint(0, 1000))
    data: DataLoader = ValidationOrTestDataset(config.validation_list, 
                                               config.device).get_data_loader(32)
    mrstft: MrstftLoss = MrstftLoss(config.device)
    print(f"# Batch count: {data.__len__()}")

    csv_print: CsvWriterProtocol = csv.writer(sys.stdout)
    csv_print.writerow(["epoch", "batch", "metric", "value"])
    epoch: int
    path: Path
    for epoch, path in itertools.islice(checkpoints.get_all(), config.skip, None):
        Trainer.load_model(model, path)

        mrstft_total_accumulator: KahanAccumulator = KahanAccumulator()
        mrstft_sc_accumulator: KahanAccumulator = KahanAccumulator()
        mrstft_mag_accumulator: KahanAccumulator = KahanAccumulator()

        batch_index: int
        batch: DataBatch
        for batch_index, batch in enumerate(data):
            predicted: Tensor2d = model.evaluate_on(batch.reverb)
            mrstft_value: MrstftLoss.Return = mrstft(predicted, batch.rir)
            
            mrstft_total: float = float(mrstft_value["total"])
            mrstft_sc: float = float(mrstft_value["sc_loss"])
            mrstft_mag: float = float(mrstft_value["mag_loss"])

            mrstft_total_accumulator.add(mrstft_total)
            mrstft_sc_accumulator.add(mrstft_sc)
            mrstft_mag_accumulator.add(mrstft_mag)

            csv_print.writerow([epoch, batch_index, "mrstft_total", mrstft_total])
            csv_print.writerow([epoch, batch_index, "mrstft_sc", mrstft_sc])
            csv_print.writerow([epoch, batch_index, "mrstft_mag", mrstft_mag])

        csv_print.writerow([epoch, "total", "mrstft_total", mrstft_total_accumulator.value()])
        csv_print.writerow([epoch, "total", "mrstft_sc", mrstft_sc_accumulator.value()])
        csv_print.writerow([epoch, "total", "mrstft_mag", mrstft_mag_accumulator.value()])

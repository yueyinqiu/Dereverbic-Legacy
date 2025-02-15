from shared.i import *
import validate_ricbe_config as config

print("# Loading...")
with torch.no_grad():
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    model: RicbeModel = RicbeModel(config.device)
    data: DataLoader = ValidationOrTestDataset(config.validation_list, 
                                               config.device).get_data_loader(32)
    print(f"# Batch count: {data.__len__()}")

    scores: dict[int, float] = {}

    csv_print: CsvWriterProtocol = csv.writer(sys.stdout)
    csv_print.writerow(["epoch", "batch", "metric", "value"])
    epoch: int
    path: Path
    for epoch, path in checkpoints.get_all():
        if epoch < config.start_checkpoint:
            continue
        
        Trainer.load_model(model, path)

        loss_total_accumulator: KahanAccumulator = KahanAccumulator()
        loss_rir_accumulator: KahanAccumulator = KahanAccumulator()
        loss_speech_accumulator: KahanAccumulator = KahanAccumulator()

        batch_index: int
        batch: DataBatch
        for batch_index, batch in enumerate(data):
            predicted: RicbeModel.Prediction = model.evaluate_on(batch.reverb)
            
            loss_l1_rir: Tensor0d = model.l1(predicted.rir, batch.rir)
            loss_stft_rir: Tensor0d = model.spec_loss(predicted.rir, batch.rir).total()
            loss_l1_speech: Tensor0d = model.l1(predicted.speech, batch.speech)
            loss_stft_speech: Tensor0d = model.spec_loss(predicted.speech, batch.speech).total()
            loss_rir: float = float(loss_l1_rir + loss_stft_rir)
            loss_speech: float = float(loss_l1_speech + loss_stft_speech)
            loss_total: float = float(loss_rir + loss_speech)

            loss_total_accumulator.add(loss_total)
            loss_rir_accumulator.add(loss_rir)
            loss_speech_accumulator.add(loss_speech)

            csv_print.writerow([epoch, batch_index, "mrstft_total", loss_total])
            csv_print.writerow([epoch, batch_index, "loss_rir", loss_rir])
            csv_print.writerow([epoch, batch_index, "loss_speech", loss_speech])

        score: float = loss_total_accumulator.value() / data.__len__()
        scores[epoch] = score

        csv_print.writerow([epoch, "average", "mrstft_total", 
                            score])
        csv_print.writerow([epoch, "average", "loss_rir", 
                            loss_rir_accumulator.value() / data.__len__()])
        csv_print.writerow([epoch, "average", "loss_speech", 
                            loss_speech_accumulator.value() / data.__len__()])

    csfile.write_all_lines(checkpoints.get_path(None) / "validation_rank.txt", 
                           [str(key) for key in sorted(scores.keys(), key=lambda key: scores[key])])
    
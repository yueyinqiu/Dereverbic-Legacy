from shared.i import *


def test(model: RicbeModel, 
         checkpoints: CheckpointsDirectory, 
         data: DataLoader, 
         criterions: list[Callable[[DataBatch, RicbeModel.Prediction], dict[str, float]]]):
    with torch.no_grad():
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

        accumulators: dict[str, KahanAccumulator] | None = None

        batch_index: int
        batch: DataBatch
        for batch_index, batch in enumerate(data):
            predicted: RicbeModel.Prediction = model.evaluate_on(batch.reverb)

            losses: dict[str, float] = {}
            criterion: Callable[[DataBatch, RicbeModel.Prediction], dict[str, float]]
            for criterion in criterions:
                losses.update(criterion(batch, predicted))

            if accumulators is None:
                accumulators = {key: KahanAccumulator() for key in losses}
            
            key: str
            for key in losses:
                csv_print.writerow([batch_index, key, losses[key]])
                accumulators[key].add(losses[key])

        assert accumulators is not None
        for key in accumulators:
            csv_print.writerow(["average", key, accumulators[key].value() / len(data)])


def main():
    import test_ricbe_config as config

    print("# Loading...")
    mrstft_ricbe: MrstftLoss = MrstftLoss(config.device, 
                                          fft_sizes=[512, 1024, 2048, 4096], 
                                          hop_sizes=[50, 120, 240, 480], 
                                          win_lengths=[512, 1024, 2048, 4096],
                                          window="hann_window")
    mrstft_fins: MrstftLoss = MrstftLoss(config.device, 
                                         fft_sizes=[i * 16000 // 48000 for i in [64, 512, 2048, 8192]],
                                         hop_sizes=[i * 16000 // 48000 for i in [32, 256, 1024, 4096]],
                                         win_lengths=[i * 16000 // 48000 for i in [64, 512, 2048, 8192]],
                                         window="hann_window")
    
    def criterion_rir_mrstft(actual: DataBatch, 
                             predicted: RicbeModel.Prediction) -> dict[str, float]:
        values: MrstftLoss.Return = mrstft_ricbe(actual.rir, predicted.rir)
        return {
            "rir_mrstft": float(values.total()),
            "rir_mrstft_mag": float(values.mag_loss),
            "rir_mrstft_sc": float(values.sc_loss),
        }
    def criterion_speech_mrstft(actual: DataBatch, 
                             predicted: RicbeModel.Prediction) -> dict[str, float]:
        values: MrstftLoss.Return = mrstft_ricbe(actual.speech, predicted.speech)
        return {
            "speech_mrstft": float(values.total()),
            "speech_mrstft_mag": float(values.mag_loss),
            "speech_mrstft_sc": float(values.sc_loss),
        }
    def criterion_rir_mrstft_fins(actual: DataBatch, 
                                  predicted: RicbeModel.Prediction) -> dict[str, float]:
        values: MrstftLoss.Return = mrstft_fins(actual.rir, predicted.rir)
        return {
            "rir_mrstft_fins": float(values.total()),
            "rir_mrstft_fins_mag": float(values.mag_loss),
            "rir_mrstft_fins_sc": float(values.sc_loss),
        }
    def criterion_reverberation_time(actual: DataBatch, 
                                     predicted: RicbeModel.Prediction) -> dict[str, float]:
        a_edc: Tensor2d = RirAcousticFeatureExtractor.energy_decay_curve_decibel(actual.rir)
        a_rt30: Tensor1d = RirAcousticFeatureExtractor.get_reverberation_time_2d(a_edc, 
                                                                                 sample_rate=16000)
        p_edc: Tensor2d = RirAcousticFeatureExtractor.energy_decay_curve_decibel(predicted.rir)
        p_rt30: Tensor1d = RirAcousticFeatureExtractor.get_reverberation_time_2d(p_edc, 
                                                                                 sample_rate=16000)
        return {"rt30": float(torch.nn.functional.mse_loss(p_rt30, a_rt30))}

    test(RicbeModel(config.device), 
         CheckpointsDirectory(config.checkpoints_directory), 
         ValidationOrTestDataset(config.test_list, config.device).get_data_loader(32),
         [
             criterion_rir_mrstft, 
             criterion_rir_mrstft_fins, 
             criterion_speech_mrstft,
             criterion_reverberation_time
         ])


if __name__ == "__main__":
    main()
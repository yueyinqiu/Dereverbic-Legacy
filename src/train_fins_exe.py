from shared.imports import *


def train(checkpoints: CheckpointsDirectory, 
          rir_train_data: WavPtDataProvider, 
          rir_validation_data: WavPtDataProvider, 
          speech_train_data: WavPtDataProvider, 
          speech_validation_data: WavPtDataProvider,
          epoch_count: int,
          model: RicModule,
          validation_interval: int,
          checkpoint_interval: int):
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters()) # type: ignore

    epoch: int = 0

    def save_checkpoint():
        path: Path = checkpoints.get_path(epoch)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rir_train_data": rir_train_data.state_dict(),
            "rir_validation_data": rir_validation_data.state_dict(),
            "speech_train_data": speech_train_data.state_dict(),
            "speech_validation_data": speech_validation_data.state_dict()
        }, path)

    latest_checkpoint: tuple[int, Path] | None = checkpoints.get_latest()
    if latest_checkpoint is None:
        save_checkpoint()
    else:
        checkpoint_path: Path
        epoch, checkpoint_path = latest_checkpoint
        checkpoint: Any = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rir_train_data.load_state_dict(checkpoint["rir_train_data"])
        rir_validation_data.load_state_dict(checkpoint["rir_validation_data"])
        speech_train_data.load_state_dict(checkpoint["speech_train_data"])
        speech_validation_data.load_state_dict(checkpoint["speech_validation_data"])
    
    model.train()
    while epoch < epoch_count or epoch_count < 0:
        epoch += 1
        optimizer.zero_grad()

        rir_batch: Tensor = rir_train_data.next_batch()
        speech_batch: Tensor = speech_train_data.next_batch()
        reverb_batch: Tensor = rir_convolve_fft.get_reverb(speech_batch, rir_batch)

        predicted: Tensor = model(speech_batch, reverb_batch)
        loss: Tensor = torch.nn.functional.mse_loss(rir_batch, predicted)
        loss.backward()
        optimizer.step()
        print(f"{epoch}: {loss:.5e}")

        if epoch % checkpoint_interval == 0:
            save_checkpoint()
            print(f"Checkpoint saved at {epoch}.")

        if epoch % validation_interval == 0:
            with torch.no_grad():
                model.eval()
                losses: list[Tensor] = []
                validation_epoch: int = 0
                for rir_batch, speech_batch in zip(rir_validation_data.iterate_all_no_random(), 
                                                   speech_validation_data.iterate_all_no_random()):
                    reverb_batch = rir_convolve_fft.get_reverb(speech_batch, rir_batch)
                    predicted = model(speech_batch, reverb_batch)
                    loss = torch.nn.functional.mse_loss(rir_batch, predicted)
                    losses.append(loss)
                    print(f"Validation {validation_epoch} at {epoch}: {loss:.5e}")
                    validation_epoch += 1
                model.train()
            print(f"Validation at {epoch}: {torch.stack(losses).mean():.5e}")


def main():
    import train_fins_config as config

    random: Random = Random(config.random_seed)
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)

    def rir_filter(x: Tensor, _):
        return x

    def speech_filter(x: Tensor, random: torch.Generator | None):
        difference: int = len(x) - config.speech_length
        if difference < 0:
            return None
        if difference == 0:
            return x
        
        if random is not None:
            start: Tensor | int = torch.randint(0, difference, tuple(), generator=random)
        else:
            start = difference // 2
        return x[start:(start + config.speech_length)]

    rir_train: WavPtDataProvider = WavPtDataProvider(config.rir_train_contents, 
                                                     rir_filter,
                                                     config.batch_size,
                                                     config.device,
                                                     random.randint(0, 1000))
    rir_validation: WavPtDataProvider = WavPtDataProvider(config.rir_validation_contents, 
                                                          rir_filter,
                                                          config.batch_size,
                                                          config.device,
                                                          random.randint(0, 1000))
    speech_train: WavPtDataProvider = WavPtDataProvider(config.speech_train_contents, 
                                                        speech_filter,
                                                        config.batch_size,
                                                        config.device,
                                                        random.randint(0, 1000))
    speech_validation: WavPtDataProvider = WavPtDataProvider(config.speech_train_contents, 
                                                             speech_filter,
                                                             config.batch_size,
                                                             config.device,
                                                             random.randint(0, 1000))
    
    model: RicModule = RicModule(config.rir_length).to(config.device)

    train(checkpoints, 
          rir_train, 
          rir_validation,
          speech_train, 
          speech_validation,
          config.epoch_count,
          model,
          config.validation_interval,
          config.checkpoint_interval)



main()

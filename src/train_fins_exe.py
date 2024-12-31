from shared import *


def train(checkpoints: CheckpointsDirectory, 
          train_data: TrainDataProvider, 
          validation_data: DataLoader, 
          batch_count: int,
          model: RicModule,
          checkpoint_interval: int):
    optimizer: AdamW = AdamW(model.parameters())

    batch_index: int = 0

    def save_checkpoint():
        path: Path = checkpoints.get_path(batch_index)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data": train_data.state_dict()
        }, path)

    latest_checkpoint: tuple[int, Path] | None = checkpoints.get_latest()
    if latest_checkpoint is None:
        save_checkpoint()
    else:
        checkpoint_path: Path
        batch_index, checkpoint_path = latest_checkpoint
        checkpoint: Any = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_data.load_state_dict(checkpoint["data"])
    
    model.train()
    while batch_index < batch_count or batch_count == -1:
        batch_index += 1
        optimizer.zero_grad()

        rir_batch: Tensor
        speech_batch: Tensor
        reverb_batch: Tensor
        rir_batch, speech_batch, reverb_batch = train_data.next_batch()

        predicted: Tensor = model(speech_batch, reverb_batch)
        loss: Tensor = torch.nn.functional.mse_loss(rir_batch, predicted)
        loss.backward()
        optimizer.step()
        print(f"{batch_index}: {loss:.5e}")

        if batch_index % checkpoint_interval == 0:
            save_checkpoint()
            print(f"Checkpoint saved at {batch_index}.")
            
            with torch.no_grad():
                model.eval()
                losses: list[Tensor] = []
                validation_epoch: int = 0
                for rir_batch, speech_batch, reverb_batch in validation_data:
                    predicted = model(speech_batch, reverb_batch)
                    loss = torch.nn.functional.mse_loss(rir_batch, predicted)
                    losses.append(loss)
                    print(f"Validation {validation_epoch} at {batch_index}: {loss:.5e}")
                    validation_epoch += 1
                print(f"Validation at {batch_index}: {torch.stack(losses).mean():.5e}")
                model.train()


def main():
    import train_fins_config as config

    random: Random = Random(config.random_seed)
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)

    train_data: TrainDataProvider = TrainDataProvider(config.train_list_rir, 
                                                      config.train_list_speech,
                                                      128,
                                                      config.device,
                                                      random.randint(0, 1000))
    validation_data: DataLoader = ValidationOrTestDataset(config.validation_list, 
                                                          config.device).get_data_loader(128)
    
    model: RicModule = RicModule().to(config.device)

    train(checkpoints, 
          train_data, 
          validation_data,
          config.batch_count,
          model,
          config.checkpoint_interval)


main()

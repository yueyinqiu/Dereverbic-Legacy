from .imports import *
from .checkpoints_directory import CheckpointsDirectory
from .data_provider import TrainDataProvider, ValidationOrTestDataset
from .rir_blind_estimation_model import RirBlindEstimationModel
import _csv

def train(checkpoints: CheckpointsDirectory, 
          train_data: TrainDataProvider, 
          model: RirBlindEstimationModel,
          checkpoint_interval: int):
    start_time: float = time.time()

    batch_index: int = 0

    def save_checkpoint():
        path: Path = checkpoints.get_path(batch_index)
        torch.save({
            "model": model.get_state(),
            "data": train_data.state_dict()
        }, path)

    latest_checkpoint: tuple[int, Path] | None = checkpoints.get_latest()
    if latest_checkpoint is None:
        save_checkpoint()
    else:
        checkpoint_path: Path
        batch_index, checkpoint_path = latest_checkpoint
        checkpoint: Any = torch.load(checkpoint_path, weights_only=True)
        model.set_state(checkpoint["model"])
        train_data.load_state_dict(checkpoint["data"])
    
    batch_index += 1
    rir_batch: Tensor
    speech_batch: Tensor
    reverb_batch: Tensor
    rir_batch, speech_batch, reverb_batch = train_data.next_batch()
    details: dict[str, float] = model.train_on(reverb_batch, rir_batch, speech_batch)
    detail_keys: list[str] = list(details.keys())
    
    print_csv: '_csv._writer' = csv.writer(sys.stdout)
    print_csv.writerow(("time", *detail_keys))
    print_csv.writerow((time.time() - start_time, *(details[key] for key in detail_keys)))
    sys.stdout.flush()

    while True:
        batch_index += 1
        rir_batch, speech_batch, reverb_batch = train_data.next_batch()
        details = model.train_on(reverb_batch, rir_batch, speech_batch)
        print_csv.writerow((time.time() - start_time, *(details[key] for key in detail_keys)))
        sys.stdout.flush()

        if batch_index % checkpoint_interval == 0:
            save_checkpoint()
            print(f"# Checkpoint saved at {batch_index}.")
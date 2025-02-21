from csv_accessing.csv_writer import CsvWriter
from data_providing.train_data_provider import TrainDataProvider
from .i0 import *
from ..checkpointing.checkpoints_directory import CheckpointsDirectory
from .rir_blind_estimation_model import RirBlindEstimationModel
from .static_class import StaticClass


class Trainer(StaticClass):
    @classmethod
    def load_model(cls,
                   model: RirBlindEstimationModel,
                   checkpoint: Path):
        checkpoint_content: Any = torch.load(checkpoint, weights_only=True)
        model.set_state(checkpoint_content["model"])

    @classmethod
    def train(cls,
              checkpoints: CheckpointsDirectory, 
              train_data: TrainDataProvider, 
              model: RirBlindEstimationModel,
              checkpoint_interval: int):
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
        rir_batch: Tensor2d
        speech_batch: Tensor2d
        reverb_batch: Tensor2d
        rir_batch, speech_batch, reverb_batch = train_data.next_batch()
        details: dict[str, float] = model.train_on(reverb_batch, rir_batch, speech_batch)
        
        print_csv: CsvWriter = csv.writer(sys.stdout)
        detail_keys: list[str] = list(details.keys())
        def print_details():
            print_csv.writerow((batch_index, 
                                time.time(), 
                                *(details[key] for key in detail_keys)))
            sys.stdout.flush()
        print_csv.writerow(("batch", "time", *detail_keys))
        print_details()

        while True:
            batch_index += 1
            rir_batch, speech_batch, reverb_batch = train_data.next_batch()
            details = model.train_on(reverb_batch, rir_batch, speech_batch)
            print_details()
            if batch_index % checkpoint_interval == 0:
                save_checkpoint()
                print(f"# Checkpoint saved at {batch_index}.")
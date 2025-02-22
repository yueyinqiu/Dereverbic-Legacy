from random import Random
from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.train_data_provider import TrainDataProvider
from models.ricbe_models.ricbe_model import RicbeModel
import train_ricbe_config
from trainers.trainer import Trainer

print("# Loading...")
random: Random = Random(train_ricbe_config.random_seed)

checkpoints: CheckpointsDirectory = CheckpointsDirectory(train_ricbe_config.checkpoints_directory)
print(f"# Checkpoints: {checkpoints.get_path(None)}")

train_data: TrainDataProvider = TrainDataProvider(train_ricbe_config.train_list_rir, 
                                                  train_ricbe_config.train_list_speech,
                                                  32,
                                                  train_ricbe_config.device,
                                                  random.randint(0, 1000))

model: RicbeModel = RicbeModel(train_ricbe_config.device)

Trainer.train(checkpoints, train_data, model, train_ricbe_config.checkpoint_interval)

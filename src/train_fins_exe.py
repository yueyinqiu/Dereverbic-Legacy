from random import Random
from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.train_data_provider import TrainDataProvider
from models.fins_models.fins_model import FinsModel
import train_fins_config
from trainers.trainer import Trainer

print("# Loading...")
random: Random = Random(train_fins_config.random_seed)

checkpoints: CheckpointsDirectory = CheckpointsDirectory(train_fins_config.checkpoints_directory)
print(f"# Checkpoints: {checkpoints.get_path(None)}")

train_data: TrainDataProvider = TrainDataProvider(train_fins_config.train_list_rir, 
                                                  train_fins_config.train_list_speech,
                                                  32,
                                                  train_fins_config.device,
                                                  random.randint(0, 1000))

model: FinsModel = FinsModel(train_fins_config.device, random.randint(0, 1000))

Trainer.train(checkpoints, train_data, model, train_fins_config.checkpoint_interval)

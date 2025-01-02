from shared import *
import train_fins_config as config

print("Loading...")
random: Random = Random(config.random_seed)
checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
train_data: TrainDataProvider = TrainDataProvider(config.train_list_rir, 
                                                  config.train_list_speech,
                                                  128,
                                                  config.device,
                                                  random.randint(0, 1000))
model: FinsModel = FinsModel(config.device, random.randint(0, 1000))
trainer.train(checkpoints, train_data, model, config.checkpoint_interval)

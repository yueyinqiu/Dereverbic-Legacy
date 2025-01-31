from shared.i import *
import train_ricbe_config as config

print("# Loading...")
random: Random = Random(config.random_seed)

checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
print(f"# Checkpoints: {checkpoints.get_path(None)}")

train_data: TrainDataProvider = TrainDataProvider(config.train_list_rir, 
                                                  config.train_list_speech,
                                                  32,
                                                  config.device,
                                                  random.randint(0, 1000))

model: RicbeModel = RicbeModel(config.device, random.randint(0, 1000))

Trainer.train(checkpoints, train_data, model, config.checkpoint_interval)

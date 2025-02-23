from random import Random

from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.train_data_provider import TrainDataProvider
from models.cleanunet_models.cleanunet_model import CleanunetModel
from trainers.trainer import Trainer


def main():
    from exe.cleanunet import train_cleanunet_config as config

    print("# Loading...")
    random: Random = Random(config.random_seed)

    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    print(f"# Checkpoints: {checkpoints.get_path(None)}")

    train_data: TrainDataProvider = TrainDataProvider(config.train_list_rir, 
                                                      config.train_list_speech,
                                                      8,
                                                      config.device,
                                                      random.randint(0, 1000))

    model: CleanunetModel = CleanunetModel(config.device)

    Trainer.train(checkpoints, train_data, model, config.checkpoint_interval)


if __name__ == "__main__":
    main()
    
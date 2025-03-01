from random import Random

from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.train_data_provider import TrainDataProvider
from models.ricbe_models.ricbe_dbe_model import RicbeDbeModel
from trainers.trainer import Trainer


def main():
    from exe.ricbe.dbe import train_ricbe_dbe_config as config

    print("# Loading...")
    random: Random = Random(config.random_seed)

    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    print(f"# Checkpoints: {checkpoints.get_path(None)}")

    train_data: TrainDataProvider = TrainDataProvider(config.train_list_rir, 
                                                      config.train_list_speech,
                                                      32,
                                                      config.device,
                                                      random.randint(0, 1000))

    model: RicbeDbeModel = RicbeDbeModel(config.device)

    Trainer.train(checkpoints, train_data, model, config.checkpoint_interval)


if __name__ == "__main__":
    main()
    
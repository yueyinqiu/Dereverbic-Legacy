from random import Random

from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.train_data_provider import TrainDataProvider
from models.cleanunet_models.cleanunet_ric_model import CleanunetRicModel
from models.ricbe_models.tdunet_ric_model import TdunetRicModel
from trainers.checkpoint_policies.checkpoint_at_interval_policy import CheckpointAtIntervalPolicy
from trainers.checkpoint_policies.checkpoint_best_after_policy import CheckpointBestAfterPolicy
from trainers.trainer import Trainer


def main():
    from exe.modified_cleanunet.ric import train_cleanunet_ric_config as config

    print("# Loading...")
    random: Random = Random(config.random_seed)

    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    print(f"# Checkpoints: {checkpoints.get_path(None)}")

    train_data: TrainDataProvider = TrainDataProvider(config.train_list_rir, 
                                                      config.train_list_speech,
                                                      32,
                                                      config.device,
                                                      random.randint(0, 1000))

    model: CleanunetRicModel = CleanunetRicModel(config.device)

    Trainer.train(checkpoints, train_data, model, 
                  CheckpointAtIntervalPolicy(config.checkpoint_interval) |
                  CheckpointBestAfterPolicy("loss_energy_decay", 10000, 0.3))


if __name__ == "__main__":
    main()
    
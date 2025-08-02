from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from models.cleanunet_models.cleanunet_dbe_model import CleanunetDbeModel
from models.ricbe_models.tdunet_dbe_model import TdunetDbeModel
from trainers.trainer import Trainer


def main():
    from exe.modified_cleanunet.dbe import validate_cleanunet_dbe_config as config
    
    print("# Loading...")
    
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    print(f"# Checkpoints: {checkpoints.get_path(None)}")

    data: ValidationOrTestDataset = ValidationOrTestDataset(
        config.validation_list, config.device)
    model: CleanunetDbeModel = CleanunetDbeModel(config.device)

    Trainer.validate(checkpoints, data.get_data_loader(32), model, config.start_checkpoint)


if __name__ == "__main__":
    main()

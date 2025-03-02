from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from models.ricbe_models.ricbe_dbe_model import RicbeDbeModel
from trainers.trainer import Trainer


def main():
    from exe.ricbe.dbe import validate_ricbe_dbe_config as config
    
    print("# Loading...")
    
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    print(f"# Checkpoints: {checkpoints.get_path(None)}")

    data: ValidationOrTestDataset = ValidationOrTestDataset(
        config.validation_list, config.device)
    model: RicbeDbeModel = RicbeDbeModel(config.device)

    Trainer.validate(checkpoints, data.get_data_loader(32), model, config.start_checkpoint)


if __name__ == "__main__":
    main()

from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from models.dereverbic_models.tdunet_ric_model_without_energy_decay import TdunetRicModelWithoutEnergyDecay
from trainers.trainer import Trainer


def main():
    from exe.ricbe.ric_without_energy_decay import validate_tdunet_ric_without_energy_decay_config as config
    
    print("# Loading...")
    
    checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
    print(f"# Checkpoints: {checkpoints.get_path(None)}")

    data: ValidationOrTestDataset = ValidationOrTestDataset(
        config.validation_list, config.device)
    model: TdunetRicModelWithoutEnergyDecay = TdunetRicModelWithoutEnergyDecay(config.device)

    Trainer.validate(checkpoints, data.get_data_loader(32), model, config.start_checkpoint)


if __name__ == "__main__":
    main()

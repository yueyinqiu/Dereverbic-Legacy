# Room Impulse Response Inverse Convolution and Blind Estimation Based on Deep Learning 

**The associated manuscript is still under development and has not been submitted yet. This repository will be updated as the research progresses.**

This repository contains the implementation of the Ricbe (Room Impulse Response Inverse Convolution and Blind Estimation) model, a deep learning-based approach for blind estimation of room impulse responses (RIRs) from reverberant speech. Ricbe aims to provide a low-cost, practical solution for RIR measurement, overcoming limitations of traditional methods.

To cite this:

```
@misc{ricbe,
  author       = {Lu, Jiaqi and Shen, Ying},
  title        = {Room Impulse Response Inverse Convolution and Blind Estimation Based on Deep Learning},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  version      = 20250617,
  doi          = {10.5281/zenodo.15679580},
  url          = {https://doi.org/10.5281/zenodo.15679580},
}
```

## Highlights

- Dereverberation + Inverse Convolution: Simplifies the complex blind estimation task into two subtasks.
- Advanced Architecture: Combines convolutional layers and LSTM layers to extract audio features for both dereverberation and inverse convolution.
- RIR Energy Decay Loss: Custom loss function tailored for RIR characteristics, enhancing prediction accuracy.
- Superior Performance: Experiments show Ricbe outperforms existing models in key metrics.

![](./readme_img/dic.png)

## License

This project is open source under [CC BY](https://creativecommons.org/licenses/by/4.0/). However, some models, loss functions, and indicators refer to other projects. Please follow the corresponding licenses. For some of them which do not clearly state the license, please be sure to respect the rights of the original author.

## Environment

We use Anaconda on Linux to manage the environments, with CUDA 12.4. We use Visual Studio Code for developpment, and highly recommend you to use it as well.

```shell
conda create -n Ricbe python=3.12.8
conda activate Ricbe
conda install conda-forge::ffmpeg=7.1.0
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r other_requirements.txt
```

## Project Structure

To make module importing easier, we regard `src/` as the root directory. So `PYTHONPATH` should be set to `src/` , otherwise the modules may not be correctly imported. If you use Visual Studio Code, there is no need to modify it manually, since we have configured `python.envFile`. However, due to the usage of relative paths, please make sure the current directory is at `Ricbe--RirBlindEstimation` when running script in a shell.

We put all executable scripts under `exe`. Other scripts will be imported by the executables to complete the tasks. Some of them can also be standalone executed and got some outputs, but it's mainly for testing purpose.

Besides them, each executable script will have a correspondig `_config.py` to store the configurations. Using python scripts for configuration allows more complicated options, and can easily be reused and located. If you sue Visual Studio Code, those configurations will be nested instead of being displayed as a standalone file.

## Datasets

### RIRs

Bird： https://github.com/FrancoisGrondin/BIRD

You could use `download_bird_exe` to download it.

### Speeches

Ears： https://github.com/facebookresearch/ears_dataset

You could use `download_ears_exe` to download it.

# RICBE 

This repository contains the implementation of the RICBE model, a deep learning-based approach for blind estimation of room impulse responses (RIRs) from reverberant speech.

## Highlights

- Dereverberation Followed By Inverse Convolution Framework
- RIC Module
- RIR Energy Decay Loss

![](./readme_img/dic.png)

## License

This project is open source under [CC BY](https://creativecommons.org/licenses/by/4.0/). However, some models, loss functions, and indicators refer to other projects. Please follow the corresponding licenses. For some of them which do not clearly state the license, please be sure to respect the rights of the original author.

## Environment

Here listed the software and hardware environment we use, but it should work on other environments including Windows or even other GPUs.

### Software

We use Anaconda on Linux (Anolis OS 8.6) to manage the environments, with CUDA 12.4. We use Visual Studio Code for developpment, and highly recommend you to use it as well.

```shell
conda create -n Ricbe python=3.12.8
conda activate Ricbe
conda install conda-forge::ffmpeg=7.1.0
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r other_requirements.txt
```

### Hardware

The hardware we use is as follows (Not all resources are occupied):

- CPU: 7x Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
    - Memory: 128800M
- GPU: 1x NVIDIA L40
    - Memory: 46068MiB

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

## Trained Models

Each model are trained for about a day and the best ones are selected based on their performance on the validation set considering their loss functions. (Actual numbers of trained epoches are suggested by their file name.)

If you don't tend to change the path of checkpoints, simply merge the existing `checkpoints/` directory and the downloaded ones.

### Ours

> THIS PART IS HIDDEN FOR DOUBLE BLIND REVIEW BUT AVAILABLE ON OPENREVIEW.

### Others

> THIS PART IS HIDDEN FOR DOUBLE BLIND REVIEW BUT AVAILABLE ON OPENREVIEW.

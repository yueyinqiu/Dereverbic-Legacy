# DeReverbIC: A TDUNET-Based Two-Stage Framework for RIR Blind Estimation

This repository contains the implementation of DeReverbIC, a deep learning-based approach for blind estimation of room impulse response (RIR) from reverberant speech.

## Environment

Here listed the software and hardware environment we use. It should work on other environments as well, including other GPUs and other operation systems, like Windows.

### Software

We use Anaconda on Linux (Anolis OS 8.6) to manage the environments, with CUDA 12.4. Visual Studio Code is used for our remote development, and is strongly recommended for anyone that would like to reproduce this project to provide a more consistent experience.

```shell
conda create -n Dereverbic python=3.12.8
conda activate Dereverbic
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

## 运行

### Project Structure

To make module importing more consistent, we regard `src/` as the root directory. So `PYTHONPATH` should be set to `src/` , otherwise the modules may not be correctly imported. If you use Visual Studio Code, there is no need to modify it manually, since we have configured `python.envFile` in the workspace. (Only take effect for the Python extension. To run in terminal, an extra set of `PYTHONPATH` is still required.)

We put all executable scripts under the `src/exe/` directory. Other scripts will be imported by these executables to complete the corresponding tasks. A `common_configurations.py` exists in `src/exe/` to allow 更方便地修改那些可能在不同设备上需要修改的设置，包括数据集和checkpoint的储存位置，以及所使用的GPU。后文指定要运行某个脚本时，简单起见，会略去 `src/exe/` 部分（这可能和其他目录混淆，如 `src/exe/data/` 和 `data/`）。

Besides them, each executable script owns a `_config.py` to store the configurations. Using python scripts for configuration allows the options be easily reused and located. If you use Visual Studio Code, those configurations will be nested accroding to the `explorer.fileNesting` configuration. 所有设置文件都尽可能和先前的配置文件相关联，例如在 `train_*_config` 修改checkpoint的储存位置后，`validate_*_config`所提供的值也会随之改变。

### 数据准备

#### RIR 数据集下载

我们使用 BIRD 数据集以提供 RIR ：https://github.com/FrancoisGrondin/BIRD

为方便起见，可以使用 `data/download/download_bird` 下载 BIRD 数据集。其储存位置由 `data/download/download_bird_config` 指定，默认为 `data/raw/bird/` 。

#### 语音数据集下载

我们使用 EARS 数据集提供干净（a开头那个词）语音： https://github.com/facebookresearch/ears_dataset

同样可以使用 `data/download/download_ears` 下载 EARS 数据集，储存位置默认为 `data/raw/ears/` 。

#### 数据预处理

下载得到的数据需要进行处理，并生成 `.wav.pt` 文件以供后续的训练和测试使用。

- RIR预处理：`data/convert_rir_to_tensor`
- 语音预处理：`data/convert_speech_to_tensor`

上述预处理中的部分步骤可能依赖于对数据集的某些特点。例如假定 RIR 数据集中所有数据长度统一，而无需进行裁剪等操作。在更换数据集后，这些预处理脚本可能不适用。

接下来我们可以使用 `data/split_dataset` 分割数据集。这个脚本是通用的，只要上述预处理脚本生成的文件符合其要求的格式。

如果需要将生成的 `wav.pt` 转换回音频文件，可以使用 `data/convert_wav_pt_to_wav`。

### 模型训练和测试

#### DeReverbIC

以我们提出的 DeReverbIC 模型为例，运行 `dereverbic/full/train_derevrbic` 即可开始训练，checkpoint默认保存到 `checkpoints/` 目录下。训练会持续运行，直到进程被关闭。

完成训练后可以使用 `dereverbic/full/validate_dereverbic` 在验证集上进行验证。验证完成后，一个 `validation_rank.txt` 会被自动存放在checkpoint的目录下，记录了各checkpoint按验证集上表现的排序。在使用 `dereverbic/full/test_dereverbic` 进行测试时，会自动使用最佳的checkpoint。验证和测试在相应数据集上运行一次后就自动结束。

#### 其他模型

其他模型的训练和测试和 DeReverbIC 是一致的。我们将所有基线模型都包括在了这个仓库中，包括在正文中涉及的：


## 术语差异

为了简化代码，部分术语可能与论文或者一般使用的术语有差异，包括但不限于以下：
- `speech` 仅指代干净语音。
- `reverb` 指代带混响语音。
- `epoch` 由于数据量非常大，每次迭代时是随机选取而不按既定的顺序，在训练时， epoch 、 batch 、iteration 的概念是混用的。

## License

This repository is open source under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). However, some models, loss functions, and indicators refer to other projects. Please follow the corresponding licenses. For some of them which do not clearly state the license, please be sure to respect the rights of the original author.

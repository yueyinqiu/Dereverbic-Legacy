# Ricbe

## 环境

我们在 Linux 上使用 Anaconda 进行环境管理。 CUDA 版本为 12.4 。

```shell
conda create -n Ricbe python=3.12.5
conda activate Ricbe
conda install ffmpeg=6.1.1
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r other_requirements.txt
```

## 项目结构

所有代码均放在 `src` 文件夹下，但执行时一般把工作目录设置在 `Ricbe--RirBlindEstimation` 处，默认配置都是按照这个工作目录设置的。

其中以 `_exe.py` 结尾的文件是设计为可以直接执行的：

```shell
python src/xxxxx_exe.py
```

而 `_exe.py` 会有与其对应的 `_config.py` 来储存配置。是的，我们直接使用 python 脚本来保存配置。这允许我们设置一些更加复杂的配置项，也能方便地复用之前的配置，并在代码中定位其配置项的使用位置。

## 数据集和预处理

### Rir 数据集

Bird： https://github.com/FrancoisGrondin/BIRD

可以使用 `download_bird_exe` 下载。

### 语音数据集

Ears： https://github.com/facebookresearch/ears_dataset

可以使用 `download_ears_exe` 下载：

### 预处理

#### 生成 wav.pt 文件

可以使用 `convert_rir_to_tensor_exe` 和 `convert_speech_to_tensor_exe` 生成 `wav.pt` 文件。

其中 rir 会同时生成一份 `wav` 文件。但发现生成 `wav` 的速度较慢，因此 speech 默认不进行生成。如果想要生成，可以在配置中打开，或者在之后使用 `convert_wav_pt_to_wav_exe` 将 `wav.pt` 转换为 `wav` 文件。

#### 切割数据集

使用 `split_bird_rir_dataset_exe` 和 `split_ears_speech_dataset_exe` 进行数据集切割。

由于这一步骤和数据集本身的结构密切相关，这里的两个脚本基本只适用于 Bird 和 Ears 数据集。如果要使用其他数据集可能需要自己相应编写。


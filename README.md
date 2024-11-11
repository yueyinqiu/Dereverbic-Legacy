# Ricbe 基于深度学习的房间冲激响应逆卷积和盲估计

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

为了更好地维护相互间的执行顺序和依赖关系，这里列出一张顺序表：

```mermaid
flowchart TB

subgraph Rir Dataset Preparation
    download_bird
    
    download_bird ==> convert_rir_to_tensor
    
    convert_rir_to_tensor ==> globally_split_bird_rir_dataset
end

subgraph Speech Dataset Preparation
    download_ears

    download_ears ==> convert_speech_to_tensor

    convert_speech_to_tensor ==> globally_split_ears_speech_dataset

    convert_speech_to_tensor --> statistically_analyze_speech
end

    convert_rir_to_tensor --> convert_wav_pt_to_wav
    convert_speech_to_tensor --> convert_wav_pt_to_wav

subgraph Fins
    globally_split_bird_rir_dataset ==> train_fins
    globally_split_ears_speech_dataset ==> train_fins
end
```


## 数据集和预处理

### Rir 数据集

Bird： https://github.com/FrancoisGrondin/BIRD

可以使用 `download_bird_exe` 下载。

### 语音数据集

Ears： https://github.com/facebookresearch/ears_dataset

可以使用 `download_ears_exe` 下载：

### 预处理

#### 生成 wav.pt 文件

可以使用 `convert_rir_to_tensor_exe` 和 `convert_speech_to_tensor_exe` 生成 `.wav.pt` 文件。

在配置中启用 `save_wav` 可以同时保存对应的 `.wav` 音频。但保存 `.wav` 文件的速度较慢，如果只需要个别音频，建议在之后使用 `convert_wav_pt_to_wav_exe` 进行转换。

#### 分析数据集

我们使用 `statistically_analyze_speech_exe` 对音频数据集的音频长度进行了统计分析，并在之后的训练中选取了合适的长度。在使用其他数据集可能需要自己进行相应分析。

#### 全局数据集切割

使用 `globally_split_bird_rir_dataset_exe` 和 `globally_split_ears_speech_dataset_exe` 进行全局数据集切割。

在之后对任何模型的训练可能都需要重新处理数据，它们不一定会完整地使用训练集，也可能再在训练集中拆分验证集，但总是不会使用测试集中的数据。当然，这里分出的测试数据数量可能也非常庞大，并不是所有组合都会被用以测试。

由于这一步骤和数据集本身的结构密切相关，这里的两个脚本基本只适用于 Bird 和 Ears 数据集。如果要使用其他数据集可能需要自己相应编写。

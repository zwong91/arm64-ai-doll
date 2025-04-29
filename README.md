
---

# 离线本地语音助手系统（基于 Fast-Whisper、MiniMind2 和 Sherpa-ONNX）

本系统支持：

- ✅ 流式关键词唤醒（KWS）
- ✅ 流式语音识别（ASR）
- ✅ 文本转语音（TTS）
- ✅ 热词增强（Hotwords）
- ✅ 全程离线运行（无需互联网）
- ✅ 无需 GPU，建议使用 **≥3TOPS 的 TPU**
- ✅ 适合部署在 **边缘侧 / 用户侧设备**

---

## 🧠 MiniMind2 Series 模型训练流程

MiniMind2 系列模型使用约 20GB 语料（≈ 4B tokens）训练完成。

### 1. 📘 预训练（Pretrain）

- **输入：** 大量原始数据（表示为 💾💾💾...）
- **处理文件：** `pretrain_hq.jsonl` (~1.6G)
- **输出：** 初始模型 / 中间状态数据

---

### 2. 🧪 监督微调（Supervised Fine-Tuning - SFT）

- **输入：** 预训练阶段的模型输出
- **处理文件：**
  - `sft_512.jsonl` (~7.5G)
  - `sft_2048.jsonl` (~9G)
- **输出：** 微调后的模型

---

### 3. 🧠 基于人类反馈的强化学习（Reinforcement Learning from Human Feedback - RLHF）

- **输入：** 微调模型输出
- **处理文件：** `dpo.jsonl` (~0.9G)
- **输出：** ✅ 最终 MiniMind2 Series 模型  
  表示为：⚛️ **MiniMind2 Series**

---

## 安装
## 🐳 使用 Docker 构建 ARM64 虚拟环境（macOS M1/M2）

下面是**在 macOS M1/M2 上使用 Docker 启动 ARM64 Ubuntu 容器并搭建 Python 环境**的详细步骤，适用于你的语音助手（基于 fast-whisper、Minimind、sherpa-onnx）项目开发。

---

## ✅ 在 macOS M1/M2 上构建 ARM64 Python 环境（非 RISC-V）

---

### 1. 安装并检查 Docker 架构支持

确保 Docker 已开启 `buildx` 且支持 ARM64（默认支持）：

```bash
docker buildx ls
```

---

### 2. 拉起 ARM64 架构的 Ubuntu 容器

```bash
docker run -it --platform linux/arm64 \
  -v "$(pwd)":/mnt/ \
  arm64v8/debian:10.13-slim \
  bash


docker ps
docker exec -it <container_id> bash
docker cp 9ebe94e97c12:/root/arm64-ai-doll/xxx.zip ./
```

> ⏱ 初次拉取镜像可能稍慢，但之后速度会很快。

---

### 3. 安装必要的依赖工具链

```bash
apt update && apt install -y \
  build-essential \
  libffi-dev \
  libbz2-dev \
  libssl-dev \
  libncurses-dev \
  libreadline-dev \
  libsqlite3-dev \
  zlib1g-dev \
  liblzma-dev \
  wget \
  curl \
  zip \
  unzip \
  p7zip-full \
  git \
  git-lfs \
  make \
  gcc \
  g++ \
  python3-venv \
  ca-certificates \
  libportaudio2 \
  libportaudiocpp0 \
  portaudio19-dev

```

---

### 4. 下载并编译 Python（以 Python 3.10 为例）

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
tar -xzf Python-3.10.13.tgz
cd Python-3.10.13

./configure --enable-optimizations
make -j2
make install

### 或者 用 pyenv 安装（适用于 ARM64）
# 安装 pyenv
curl https://pyenv.run | bash

# 添加环境变量（或加进 ~/.bashrc）
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 安装 Python 3.10.13
pyenv install 3.10.13
pyenv global 3.10.13

# 检查版本
python --version

```

> ⚙️ 这一步大概几分钟，取决于你的 Mac 性能。

---

### 5. 创建并激活虚拟环境

```bash
python3.10 -m venv /opt/arm64_venv
source /opt/arm64_venv/bin/activate
```

---

### 6. 安装 Rust（用于 Minimind）

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

---

### 7. 安装项目依赖（建议使用清华源）

```bash
git clone https://github.com/zwong91/arm64-ai-doll.git
# 假设你项目已经挂载到了容器内，比如 /workspace/arm64-ai-doll
cd arm64-ai-doll

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip uninstall -y numpy
pip install "numpy<2"

```

---

### 8. 可选：打包虚拟环境备份或复制到实体设备

```bash
tar czf /arm64_venv.tar.gz /opt/arm64_venv
```

然后在宿主机另开终端复制出来：

```bash
docker ps  # 查找 container ID
docker cp96991223cfe5:/root/arm64-ai-doll/dist ./
```
---

## 使用方法

1. 文件处理模式:
```bash
python main.py -f path/to/audio.wav

python main.py --list-devices

OPENBLAS_NUM_THREADS=1 python main.py -f input.wav --output-device "蓝牙耳机"


# 如果有默认的麦克风和扬声器 参数input/output device 不填写
./arm64_ai_doll --list-devices
./arm64_ai_doll -i --input-device "麦克风<name or ID>" --output-device "扬声器<name or ID>"


apt install alsa-utils pulseaudio

python main.py -i --input-device "default" --output-device "default"

```

2. 交互式模式:
```bash
python main.py --watch-dir path/to/audio.wav --asr-model sensevoice --pid-file /path/to/pidfile.txt
kill $(cat /path/to/pidfile.txt)
```



如果你想将类似 `python main.py -f example.mp3` 的 Python 脚本打包成一个可以分发的独立软件，可以使用几种工具来创建一个可执行的程序。以下是一些常见的步骤：

### 1. 使用 PyInstaller 打包为可执行文件

**PyInstaller** 是一个将 Python 程序打包为独立可执行文件的工具。它会将 Python 解释器和所有依赖打包到一个可执行文件中，用户无需安装 Python 环境即可运行。

#### 1.1 安装 PyInstaller
首先，你需要安装 `PyInstaller`：
```bash
pip install pyinstaller

pyinstaller --clean --onedir --noupx --name arm64_ai_doll \
  --add-data "sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01:sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01" \
  --add-data "sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en:sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en" \
  --add-data "sensevoice_ckpt:sensevoice_ckpt" \
  --add-data "whisper_ckpt:whisper_ckpt" \
  --add-data "vad_ckpt:vad_ckpt" \
  --add-data "sherpa/vits-icefall-zh-aishell3:sherpa/vits-icefall-zh-aishell3" \
  --add-data "speech-enhancement:speech-enhancement" \
  --add-data "MiniMind2-Small:MiniMind2-Small" \
  --add-data "model/minimind_tokenizer:model/minimind_tokenizer" \
  --add-data "keywords:keywords" \
  main.py


(arm64_venv) root@6f2423763367:~/arm64-ai-doll# ldconfig -p | grep portaudio
	libportaudiocpp.so.0 (libc6,AArch64) => /usr/lib/aarch64-linux-gnu/libportaudiocpp.so.0
	libportaudiocpp.so (libc6,AArch64) => /usr/lib/aarch64-linux-gnu/libportaudiocpp.so
	libportaudio.so.2 (libc6,AArch64) => /usr/lib/aarch64-linux-gnu/libportaudio.so.2
	libportaudio.so (libc6,AArch64) => /usr/lib/aarch64-linux-gnu/libportaudio.so


1045 INFO: PyInstaller: 6.13.0, contrib hooks: 2025.3
1048 INFO: Python: 3.10.12
1081 INFO: Platform: Linux-6.8.0-1024-aws-aarch64-with-glibc2.35
1081 INFO: Python environment: /opt/arm64_venv
1085 INFO: wrote /root/arm64-ai-doll/arm64_ai_doll.spec
1099 INFO: Module search paths (PYTHONPATH):
['/usr/lib/python310.zip',
 '/usr/lib/python3.10',
 '/usr/lib/python3.10/lib-dynload',
 '/opt/arm64_venv/lib/python3.10/site-packages',
 '/root/arm64-ai-doll']
3350 INFO: Appending 'datas' from .spec

```

#### 1.2 创建可执行文件
然后，使用以下命令将你的 `main.py` 文件打包成一个可执行文件：
```bash
pyinstaller --clean --onedir --noupx --name arm64_ai_doll \
  --add-data "sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01:sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01" \
  --add-data "sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en:sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en" \
  --add-data "sensevoice_ckpt:sensevoice_ckpt" \
  --add-data "whisper_ckpt:whisper_ckpt" \
  --add-data "vad_ckpt:vad_ckpt" \
  --add-data "sherpa/vits-icefall-zh-aishell3:sherpa/vits-icefall-zh-aishell3" \
  --add-data "speech-enhancement:speech-enhancement" \
  --add-data "MiniMind2-Small:MiniMind2-Small" \
  --add-data "model/minimind_tokenizer:model/minimind_tokenizer" \
  --add-data "keywords:keywords" \
  main.py

```
- `--onefile` 选项会将所有文件打包成一个单独的可执行文件。
- 生成的可执行文件通常位于 `dist/` 目录下。

#### 1.3 使用命令行参数
你的 `main.py` 可以通过 `argparse` 等工具接收命令行参数。例如，修改 `main.py` 使其接受 `--file` 参数：
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process an MP3 file.")
    parser.add_argument('--file', type=str, help='Path to the MP3 file')
    args = parser.parse_args()

    # 你的处理逻辑
    print(f"Processing {args.file}...")

if __name__ == "__main__":
    main()
```

#### 1.4 分发
生成的可执行文件可以直接分发给用户，他们无需安装 Python 或依赖包。用户只需要双击或在命令行运行即可：
```bash
./arm64_ai_doll -f example.mp3 --output-dir ./outputs/
```

## 项目结构

```
src/
├── core/          # 核心功能模块
├── utils/         # 工具函数
└── config.py      # 配置文件
```

## 支持的功能

- 语音识别 (fast-whisper)
- 自然语言处理 (MiniMind)
- 语音合成 (sherpa-onnx)
- 实时录音流对话


关键词检测：https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html#sherpa-onnx-kws-zipformer-wenetspeech-3-3m-2024-01-01-chinese
本质是一个非常小的语音识别模型，这里用它来实现语音唤醒（一直监听音频流），类似小爱同学
支持自定义、复数个关键词且不需要重新训练。

```bash
sherpa-onnx-cli text2token \
  --tokens sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
  --tokens-type ppinyin \
  keywords_raw.txt keywords.txt
```

ASR：https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
在成功语音唤醒后，才会从语音唤醒模式切换到语音识别模式，对音频流进行语音识别，进行自己的业务处理
模型的识别率还可以、支持中英文混合、RTF表现优秀，支持热词。

使用热词是由于项目中有一些专业性的词汇，不设置热词的默认识别率比较低。热词也是可以自定义、复数个、不需要重新训练。
https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html

TTS： https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#aishell3-chinese-multi-speaker-174-speakers


## Q & A
  Traceback (most recent call last):
  File "/mnt/sdb/shared/sherpa-onnx/./python-api-examples/vad-microphone.py", line 8, in <module>
    import sounddevice as sd
  File "/mnt/sdb/shared/py311/lib/python3.11/site-packages/sounddevice.py", line 71, in <module>
    raise OSError('PortAudio library not found')
OSError: PortAudio library not found
Then please run:

sudo apt-get install libportaudio2

这个 `libportaudio2` 是 **PortAudio** 动态库，  
通常是系统装的，不太有直接“纯离线包”或者“官方静态版”。

不过你有方法可以搞定离线安装：

https://packages.debian.org/sid/libportaudio2

```bash
apt-get install --download-only libportaudio2

cd Offline-PortAudio
dpkg -i *.deb

```

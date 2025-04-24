---

# 语音助手

基于 fast-whisper、Minimind 和 sherpa-onnx 的语音助手系统。

## 安装
## 🐳 使用 Docker 构建 ARM64 虚拟环境（macOS M1/M2）

### ✅ 1. 安装 Docker（支持 `--platform`）

确保你已经安装了支持多架构（包括 QEMU）的 Docker 版本（Docker Desktop for Mac 是 OK 的）。

验证支持 arm aarch64：

```bash
docker buildx ls
```

---

明白了！下面是**在 macOS M1/M2 上使用 Docker 启动 ARM64 Ubuntu 容器并搭建 Python 环境**的详细步骤，适用于你的语音助手（基于 fast-whisper、Minimind、sherpa-onnx）项目开发。

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
docker run --rm -it --platform linux/arm64 arm64v8/ubuntu:22.04 bash
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
  wget \
  curl \
  git \
  make \
  gcc \
  g++ \
  python3-venv \
  ca-certificates
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
source $HOME/.cargo/env
```

---

### 7. 安装项目依赖（建议使用清华源）

```bash
# 假设你项目已经挂载到了容器内，比如 /workspace/ai-doll
cd /workspace/ai-doll

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 8. 可选：打包虚拟环境备份或复制到实体设备

```bash
tar czf /arm64_venv.tar.gz /opt/arm64_venv
```

然后在宿主机另开终端复制出来：

```bash
docker ps  # 查找 container ID
docker cp <container_id>:/arm64_venv.tar.gz ./
```

---

## 使用方法

1. 文件处理模式:
```bash
python main.py --file path/to/audio.wav
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
```

#### 1.2 创建可执行文件
然后，使用以下命令将你的 `main.py` 文件打包成一个可执行文件：
```bash
pyinstaller --onefile \
  --name arm64_ai_doll \
  --add-data "whisper_ckpt:whisper_ckpt" \
  --add-data "sensevoice_ckpt:sensevoice_ckpt" \
  --add-data "vits-icefall-zh-aishell3:vits-icefall-zh-aishell3" \
  --add-data "MiniMind2-Small:MiniMind2-Small" \
  --add-data "model/minimind_tokenizer:model/minimind_tokenizer" \
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

### models
```
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
rm vits-icefall-zh-aishell3.tar.bz2

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
- 实时录音对话
  

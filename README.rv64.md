在 Docker 里直接跑 RISC-V 64 架构的镜像对吧？这确实是个很“惊喜”的想法😂，但还是得注意一些坑。下面是关键点和一些可选方案：

---

## 🧠 你需要知道的核心点：

1. **你的宿主机是什么架构？**
   - 如果是 x86_64（比如常见的 Intel / AMD CPU），那你要跑 RISC-V 64 的 Docker 镜像，就 **必须用 QEMU 模拟器** 来实现跨架构模拟。

2. **RISC-V 镜像不多**，而且基本上都很轻量，适合嵌入式或者系统开发用途。

3. **性能会非常慢**，因为 QEMU 是全模拟（不像 ARM 的 Rosetta2 或者 Apple 的 M 系列那样可以高效做 JIT 或部分指令转译）。

---

## 🚀 快速开始方案（适合 x86_64 上跑 RISC-V Docker）

### 步骤 1：安装 QEMU 支持

```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

这一步会设置 binfmt 支持各种架构。

---

### 步骤 2：运行一个 RISC-V 的镜像

alpine（更小）：

```bash
docker run --rm -it --platform=linux/riscv64 alpine
```

---

你运行的 RISC-V 镜像没有预装 `apt` 包管理工具，这可能是因为使用的是一个极简的镜像（比如 `alpine` 或者其他类似镜像）。如果你使用的是类似 `alpine` 的基础镜像，它通常会使用 `apk` 来代替 `apt`，而且并没有预装 Python 相关的包。

### 解决方法：使用 `apk` 安装依赖

如果是 `alpine` 镜像，你需要使用 `apk` 来安装包。以下是安装 Python 环境及编译工具的步骤：

1. **安装 `apk` 包管理工具**

   首先更新 `apk` 并安装需要的包：

   ```bash
   apk update
   apk add --no-cache python3 py3-pip build-base git git-lfs curl wget pkgconfig ffmpeg-dev
   ```

   这会安装 Python 3 和相关的编译工具（`build-base` 包含了 `gcc`、`make` 等工具）。

2. **安装 Python 包**

解决方法是 **使用虚拟环境** 来安装 Python 包，这不仅符合最佳实践，还能避免污染系统的 Python 安装。

### 解决方案：使用虚拟环境

按照以下步骤创建一个 Python 虚拟环境并在其中安装依赖：

#### 1. 创建虚拟环境

```bash
python --version
python3 -m venv myenv
```

这会在当前目录下创建一个名为 `myenv` 的虚拟环境。

#### 2. 激活虚拟环境

```bash
source myenv/bin/activate
```

激活后，命令行前面会显示 `(myenv)`，表示当前正在使用虚拟环境。

#### 3. 安装依赖

在虚拟环境中，安装项目的 Python 依赖：

```bash
pip install -r requirements.txt
```

#### 4. 退出虚拟环境

安装完成后，可以通过以下命令退出虚拟环境：

```bash
deactivate
```

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```


## 5, Pytorch

目前，适用于 RISC-V 64 架构的官方 PyTorch wheel 包仍然较为稀缺，社区构建的版本可能存在兼容性问题，导致在您的环境中无法成功安装。

### 🛠️ 推荐方案：从源码编译 PyTorch
https://github.com/KumaTea/pytorch-riscv64

5. **验证安装**：

   安装完成后，您可以通过以下命令验证 PyTorch 是否安装成功：

   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```
---


## 使用方法

1. 文件处理模式:
```bash
python main.py --file path/to/audio.wav
```

2. 交互式模式:
```bash
python main.py --watch-dir test
```
如果你想将类似 `python main.py --file example.mp3` 的 Python 脚本打包成一个可以分发的独立软件，可以使用几种工具来创建一个可执行的程序。以下是一些常见的步骤：

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
  --add-data "vits-icefall-zh-aishell3:vits-icefall-zh-aishell3" \
  --add-data "MiniMind2-Small:MiniMind2-Small" \
  --add-data "model/minimind_tokenizer:model/minimind_tokenizer" \
  main.py

```
- `--onefile` 选项会将所有文件打包成一个单独的可执行文件。
- 生成的可执行文件通常位于 `dist/` 目录下。
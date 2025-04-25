#!/bin/bash

set -e

echo ">>> 1. 安装必要工具链和 PortAudio"
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
  zip \
  unzip \
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

echo ">>> 2. 安装 pyenv 并配置环境变量"
curl https://pyenv.run | bash

# 添加到当前 shell 环境（可追加到 ~/.bashrc 或 ~/.profile）
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo ">>> 3. 安装 Python 3.10.13"
pyenv install 3.10.13
pyenv global 3.10.13
python --version

echo ">>> 4. 创建并激活虚拟环境"
python3.10 -m venv /opt/arm64_venv
source /opt/arm64_venv/bin/activate

echo ">>> 5. 安装 Rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo ">>> 6. 克隆项目代码"
# git clone https://github.com/zwong91/arm64-ai-doll.git
# cd arm64-ai-doll

echo ">>> 7. 安装 Python 依赖（包括 numpy 降级）"
pip uninstall -y numpy
pip install "numpy<2"

echo ">>> 8. 下载语音合成模型 (TTS)"
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
rm vits-icefall-zh-aishell3.tar.bz2

echo ">>> 9. 下载 Sherpa ONNX 推理库"
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.3/sherpa-onnx-v1.11.3-linux-aarch64-static.tar.bz2
tar xvf sherpa-onnx-v1.11.3-linux-aarch64-static.tar.bz2
rm sherpa-onnx-v1.11.3-linux-aarch64-static.tar.bz2

echo ">>> 10. 下载 MiniMind 模型"
git lfs install
git clone https://huggingface.co/jingyaogong/MiniMind2-Small
rm -rf MiniMind2-Small/.git

echo ">>> 11. 下载 Whisper 模型并整理目录"
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-tiny
mkdir -p whisper_ckpt
mv faster-whisper-tiny/* whisper_ckpt/

echo ">>> 12. 安装 Python 项目依赖"
pip install -r requirements.txt


echo ">>> 13. 使用 PyInstaller 构建 arm64_ai_doll"
pip install pyinstaller

echo "📦 Step 14: 开始 PyInstaller 打包"
pyinstaller --clean --onedir --noupx --name arm64_ai_doll \
  --add-data "whisper_ckpt:whisper_ckpt" \
  --add-data "vits-icefall-zh-aishell3:vits-icefall-zh-aishell3" \
  --add-data "MiniMind2-Small:MiniMind2-Small" \
  --add-data "model/minimind_tokenizer:model/minimind_tokenizer" \
  --add-binary "3rd/libportaudio.a:./libportaudio" \
  --add-data "3rd/portaudio.h:." \
  --collect-binaries sounddevice \
  main.py

echo "🚀 Step 15: 运行打包后的程序"
cp -r 3rd dist/arm64_ai_doll/
cd dist/arm64_ai_doll

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
#./arm64_ai_doll -i --input-device default --output-device default

echo ">>> ✅ 构建完成，输出目录为 dist/arm64_ai_doll"

# 获取当前 Git 版本号
GIT_VER=$(git rev-parse --short HEAD)
OUTPUT_NAME="arm64_ai_doll_${GIT_VER}.zip"

echo ">>> 16. 压缩构建输出为 $OUTPUT_NAME"

# 进入 dist/
cd ..

# 打包 arm64_ai_doll 目录（保持结构），输出在 dist 目录外
zip -r "../$OUTPUT_NAME" arm64_ai_doll

# 回到原目录
cd ..

echo ">>> ✅ 压缩完成：$(realpath $OUTPUT_NAME)"
echo ">>> 📦 文件大小：$(du -sh $OUTPUT_NAME | cut -f1)"


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
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xvf kokoro-multi-lang-v1_0.tar.bz2
rm kokoro-multi-lang-v1_0.tar.bz2

echo ">>> 9. 下载 Sherpa ONNX 推理库"
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.11.3/sherpa-onnx-v1.11.3-linux-aarch64-static.tar.bz2
tar xvf sherpa-onnx-v1.11.3-linux-aarch64-static.tar.bz2
rm sherpa-onnx-v1.11.3-linux-aarch64-static.tar.bz2

pip install huggingface_hub


echo ">>> 10. 下载 MiniMind 模型"
git lfs install
git clone https://huggingface.co/jingyaogong/MiniMind2-Small
#HF_ENDPOINT=https://hf-mirror.com huggingface-cli download jingyaogong/MiniMind2-Small --local-dir MiniMind2-Small 
rm -rf MiniMind2-Small/.git

echo ">>> 11. 下载 Whisper 模型并整理目录"
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-tiny
#HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Systran/faster-whisper-tiny --local-dir faster-whisper-tiny
mkdir -p whisper_ckpt
mv faster-whisper-tiny/* whisper_ckpt/

echo ">>> 12. 安装 Python 项目依赖"
pip install -r requirements.txt


echo ">>> 13. 使用 PyInstaller 构建 arm64_ai_doll"
pip install pyinstaller


echo ">>>14. 编译安装 PortAudio"
apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    libpulse-dev

cd /tmp
wget http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar xf pa_stable_v190700_20210406.tgz
cd portaudio
./configure --enable-static --disable-shared
make
make install
ldconfig

cd -

echo "📦 Step 15: 开始 PyInstaller 打包"
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

#echo "🚀 Step 15: 运行打包后的程序"
#cd dist/arm64_ai_doll

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
#./arm64_ai_doll -i --input-device default --output-device default

echo ">>> ✅ 构建完成，输出目录为 dist/arm64_ai_doll"

# 获取当前 Git 版本号
GIT_VER=$(git rev-parse --short HEAD)
OUTPUT_NAME="arm64_ai_doll_${GIT_VER}.zip"

echo ">>> 16. 压缩构建输出为 $OUTPUT_NAME"

cp -r *.mp3 dist/arm64_ai_doll/

# 进入 dist/
cd dist

# 打包 arm64_ai_doll 目录（保持结构），输出在 dist 目录外
zip -r "../$OUTPUT_NAME" arm64_ai_doll

# 回到原目录
cd ..

echo ">>> ✅ 压缩完成：$(realpath $OUTPUT_NAME)"
echo ">>> 📦 文件大小：$(du -sh $OUTPUT_NAME | cut -f1)"


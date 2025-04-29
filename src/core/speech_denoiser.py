import os
import sys
import torch
import soundfile as sf
import sherpa_onnx
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from ..utils.utils import resource_path

class SpeechEnhancer:
    def __init__(self, model_path: str = "speech-enhancement/gtcrn_simple.onnx", device: str = "cpu", **kwargs):
        """
        初始化语音增强模型
        model_path: 模型文件的路径
        device: 使用的设备，默认为 'cpu'
        kwargs: 其他参数
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_path = resource_path(model_path)
        self.model = self._create_denoiser()

    def _create_denoiser(self) -> sherpa_onnx.OfflineSpeechDenoiser:
        """创建并返回语音增强模型"""
        if not Path(self.model_path).is_file():
            raise ValueError(f"Model file {self.model_path} not found. Please download it first.")
        
        config = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=self.model_path
                ),
                debug=False,
                num_threads=1,
                provider=self.device,
            )
        )
        
        if not config.validate():
            print(config)
            raise ValueError("Errors in config. Please check previous error logs")

        return sherpa_onnx.OfflineSpeechDenoiser(config)

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """对音频进行降噪处理"""
        start = time.time()
        audio = np.ascontiguousarray(audio)
        denoised = self.model(audio, sample_rate)
        end = time.time()

        elapsed_seconds = end - start
        audio_duration = len(audio) / sample_rate
        real_time_factor = elapsed_seconds / audio_duration

        print(f"增强耗时: {elapsed_seconds:.3f}秒")
        print(f"音频时长: {audio_duration:.3f}秒")
        print(f"实时因子: {real_time_factor:.3f}")

        return denoised.samples


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    """加载音频文件并返回样本和采样率"""
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # 只使用第一个通道（单声道）
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def save_audio(filename: str, samples: np.ndarray, sample_rate: int):
    """保存音频样本到文件"""
    sf.write(filename, samples, sample_rate)
    print(f"保存至: {filename}")


def process_audio(model_path: str, input_audio_path: str, output_audio_path: str = "./enhanced_16k.wav"):
    """主函数，加载音频，增强处理并保存结果"""
    enhancer = SpeechEnhancer(model_path=model_path)

    samples, sample_rate = load_audio(input_audio_path)
    enhanced_audio = enhancer.enhance(samples, sample_rate)
    save_audio(output_audio_path, enhanced_audio, sample_rate)


if __name__ == "__main__":
    # 示例：处理音频文件
    model_path = "./gtcrn_simple.onnx"  # 语音增强模型路径
    input_audio_path = "./speech_with_noise.wav"  # 输入音频文件路径
    output_audio_path = "./enhanced_16k.wav"  # 输出音频文件路径

    process_audio(model_path, input_audio_path, output_audio_path)

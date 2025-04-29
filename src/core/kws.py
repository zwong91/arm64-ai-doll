import sounddevice as sd
import sherpa_onnx
from pathlib import Path
from typing import Optional, List
from ..utils.utils import resource_path

import os
import platform
import torch

def detect_num_threads():
    # 根据系统 CPU 核心数来设置线程数
    return os.cpu_count()  # 获取系统的 CPU 核心数

def detect_provider():
    # 自动检测提供者：如果有 GPU 就用 CUDA，否则用 CPU
    system = platform.system().lower()
    if system == "darwin":
        return "coreml"  # 如果是 Mac，使用 coreml
    elif torch.cuda.is_available():
        return "cuda"  # 如果有 GPU，使用 CUDA
    else:
        return "cpu"  # 默认使用 CPU

class KeywordSpotter:
    def __init__(
        self,
        tokens_path: str,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        keywords_file: str,
        num_threads: int = 1,
        provider: str = "cpu",
        max_active_paths: int = 4,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        num_trailing_blanks: int = 1,
        sample_rate: int = 16000,
    ):
        # 验证文件存在性
        self._check_files_exist([tokens_path, encoder_path, decoder_path, joiner_path, keywords_file])

        num_threads = detect_num_threads()
        provider = detect_provider()

        print(f"Number of threads: {num_threads}")
        print(f"Provider: {provider}")

        # 初始化KWS配置
        self.keyword_spotter = sherpa_onnx.KeywordSpotter(
            tokens=resource_path(tokens_path),
            encoder=resource_path(encoder_path),
            decoder=resource_path(decoder_path),
            joiner=resource_path(joiner_path),
            keywords_file=resource_path(keywords_file),
            num_threads=num_threads,
            provider=provider,
            max_active_paths=max_active_paths,
            keywords_score=keywords_score,
            keywords_threshold=keywords_threshold,
            num_trailing_blanks=num_trailing_blanks,
        )
        
        self.sample_rate = sample_rate
        self.stream = self.keyword_spotter.create_stream()
        
    def _check_files_exist(self, files: List[str]):
        for file in files:
            if not Path(resource_path(file)).is_file():
                raise FileNotFoundError(f"文件不存在: {file}")

    def process_audio(self, audio_samples) -> Optional[str]:
        """处理音频数据并返回检测到的关键词"""
        self.stream.accept_waveform(self.sample_rate, audio_samples)
        
        result = None
        while self.keyword_spotter.is_ready(self.stream):
            self.keyword_spotter.decode_stream(self.stream)
            detect_result = self.keyword_spotter.get_result(self.stream)
            if detect_result:
                result = detect_result
                self.keyword_spotter.reset_stream(self.stream)
                break
        
        return result

    def reset(self):
        """重置检测流"""
        self.keyword_spotter.reset_stream(self.stream)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'stream'):
            self.reset()

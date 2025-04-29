import sys
import os
import re
import torch
from faster_whisper import WhisperModel
import sherpa_onnx

import tempfile
import soundfile as sf

from src.utils.utils import resource_path

def remove_tags(text: str) -> str:
    return re.sub(r"<\|.*?\|>", "", text)

class SpeechToText:
    def __init__(self, backend="whisper", **kwargs):
        """
        backend: 选择后端，"whisper" 或 "sensevoice"
        kwargs: 根据 backend 传不同的初始化参数
        """
        self.backend = backend.lower()
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if self.backend == "whisper":
            self._init_whisper(kwargs)
        elif self.backend == "sensevoice":
            self._init_sensevoice(kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_whisper(self, kwargs):
        model_path = resource_path(kwargs.get("model_path", "whisper_ckpt"))
        compute_type = kwargs.get("compute_type", "int8")
        self.model = WhisperModel(model_path, device=self.device, compute_type=compute_type)

    def _init_sensevoice(self, kwargs):
        model_path = resource_path(kwargs.get("model_path", "sensevoice_ckpt"))
        #self.model = AutoModel(model=model_path, trust_remote_code=True, device=self.device, disable_update=True)
        # 获取系统的 CPU 核心数
        cpu_cores = os.cpu_count()
        # 设置 num_threads 为 CPU 核心数
        num_threads = cpu_cores if cpu_cores else 1  # 如果获取失败，默认为 1
        self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_path / "model.int8.onnx"),
            tokens=str(model_path / "tokens.txt"),
            num_threads = num_threads,
            language="auto", #auto, zh, en, ko, ja, and yue
            use_itn=True,
            debug=False,
        )

    def transcribe(self, sample_rate, audio):
        if audio.ndim == 2:
            audio = audio[:, 0]

        if self.backend == "whisper":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sample_rate)
                segments, _ = self.model.transcribe(f.name)
            return " ".join([s.text for s in segments])

        elif self.backend == "sensevoice":
            stream = self.model.create_stream()
            stream.accept_waveform(sample_rate, audio)
            self.model.decode_stream(stream)
            return stream.result

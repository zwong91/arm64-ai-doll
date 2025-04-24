import sys
import os
import re
import torch
from faster_whisper import WhisperModel
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

def resource_path(path: str) -> str:
    """返回资源文件的实际路径"""
    if os.path.isabs(path):
        return path
    base_path = getattr(sys, "_MEIPASS", None) or os.path.abspath(".")
    return os.path.join(base_path, path)

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
        self.model = AutoModel(model=model_path, trust_remote_code=True, device=self.device, disable_update=True)

    def transcribe(self, audio_file):
        if self.backend == "whisper":
            segments, _ = self.model.transcribe(audio_file)
            return " ".join([segment.text for segment in segments])
        elif self.backend == "sensevoice":
            result = self.model.generate(
                input=audio_file,
                cache={},
                language="auto",
                use_itn=False,
                batch_size=64
            )[0]["text"].strip()
            return remove_tags(result)

import logging
import sys
import os
import re
import torch
import sherpa_onnx

import tempfile
import soundfile as sf

from src.utils.utils import resource_path

def remove_tags(text: str) -> str:
    return re.sub(r"<\|.*?\|>", "", text)

class SpeechToText:
    def __init__(self, backend="sensevoice", **kwargs):
        """
        backend: 选择后端，"paraformer" 或 "sensevoice"
        kwargs: 根据 backend 传不同的初始化参数
        """
        self.backend = backend.lower()
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"asr model: {backend}")
        if self.backend == "sensevoice":
            self._init_sensevoice(kwargs)
        elif self.backend == "paraformer":
            self._init_paraformer(kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_sensevoice(self, kwargs):
        model_path = resource_path(kwargs.get("model_path", "sherpa/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))
        #self.model = AutoModel(model=model_path, trust_remote_code=True, device=self.device, disable_update=True)
        # 获取系统的 CPU 核心数
        cpu_cores = os.cpu_count()
        # 设置 num_threads 为 CPU 核心数
        num_threads = cpu_cores if cpu_cores else 1  # 如果获取失败，默认为 1
        self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=resource_path(os.path.join(model_path, "model.int8.onnx")),
            tokens=resource_path(os.path.join(model_path, "tokens.txt")),
            num_threads=num_threads,
            language="auto",
            use_itn=True,
            debug=False,
        )

    def _init_paraformer(self, kwargs):
        model_path = resource_path(kwargs.get("model_path", "sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en"))
        encoder = os.path.join(model_path, "encoder.int8.onnx")
        decoder = os.path.join(model_path, "decoder.int8.onnx")
        tokens = os.path.join(model_path, "tokens.txt")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=resource_path(tokens),
            encoder=resource_path(encoder),
            decoder=resource_path(decoder),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,  # it essentially disables this rule
        )

    def transcribe(self, sample_rate, audio):
        if self.backend == "whisper":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sample_rate)
                segments, _ = self.model.transcribe(f.name)
            return " ".join([s.text for s in segments])

        elif self.backend == "sensevoice":
            stream = self.model.create_stream()
            stream.accept_waveform(sample_rate, audio)
            self.model.decode_stream(stream)
            return stream.result.text.strip()

        elif self.backend == "paraformer":
            # 实时语音识别
            last_result = ""
            segment_id = 0
            results = []
            
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            is_endpoint = self.recognizer.is_endpoint(stream)

            result = self.recognizer.get_result(stream)
            debug = False
            if result and (last_result != result):
                last_result = result
                if debug: logging.info("\r{}:{}".format(segment_id, result))
            if is_endpoint:
                if result:
                    if debug:logging.info("\r{}:{}".format(segment_id, result))
                    segment_id += 1
                    # generator result
                    #yield result
                    results.append(result)
                self.recognizer.reset(stream)
            return " ".join(results)       
import os
import sys
import queue
import threading
import time
import logging

import numpy as np
import sherpa_onnx
import soundfile as sf
import sounddevice as sd

from ..utils.utils import resource_path
from .share_state import State

buffer = queue.Queue()
started = False
stopped = False
killed = False
sample_rate = None
event = threading.Event()
first_message_time = None
play_thread_started = False
play_thread_lock = threading.Lock()


def generated_audio_callback(samples: np.ndarray, progress: float):
    global started, first_message_time
    if first_message_time is None:
        first_message_time = time.time()
    buffer.put(samples)
    if not started:
        logging.info("Start playing ...")
        started = True
        State.pause_listening() # 禁用监听
    return 0 if killed else 1


def play_audio_callback(outdata: np.ndarray, frames: int, cbtime, status: sd.CallbackFlags):
    if killed:
        event.set()

    if buffer.empty():
        outdata.fill(0)
        State().resume_listening()  # 启用监听
        return

    n = 0
    while n < frames and not buffer.empty():
        remaining = frames - n
        k = buffer.queue[0].shape[0]

        if remaining <= k:
            outdata[n:, 0] = buffer.queue[0][:remaining]
            buffer.queue[0] = buffer.queue[0][remaining:]
            n = frames
            if buffer.queue[0].shape[0] == 0:
                buffer.get()
            break

        outdata[n : n + k, 0] = buffer.get()
        n += k

    if n < frames:
        outdata[n:, 0] = 0


def play_audio():
    with sd.OutputStream(
        channels=1,
        callback=play_audio_callback,
        dtype="float32",
        samplerate=sample_rate,
        blocksize=4096,
        latency='high',  # 或 0.1
    ):
        event.wait()
    logging.info("Exiting ...")


def stop_playback():
    global killed
    killed = True
    State().resume_listening()
    event.set()


class TextToSpeech:
    def __init__(self, 
                 model_dir="sherpa/vits-icefall-zh-aishell3",
                 output_device=None,  # 默认输出设备
                 backend="sherpa-onnx",
                 voice="af_alloy",   
                 speed=1.3,
        ):
        self.backend = backend
        self.voice = voice
        self.speed = speed
        self.output_device = int(output_device) if output_device.isdigit() else output_device
        real_path = resource_path(model_dir)
        if real_path is None:
            raise ValueError("model_dir must be specified")
        self.model_dir = real_path
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def synthesize(self, text):
        if self.backend == "sherpa-onnx":
            self._synthesize_sherpa_onnx(text)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _synthesize_sherpa_onnx(self, text):
        import torch
        import platform

        def detect_provider():
            system = platform.system().lower()
            if system == "darwin":
                return "coreml"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        try:
            model_files = {
                "model": None,
                "lexicon": None,
                "tokens": None,
                "dict_dir": None,
                "rule_fsts": [],
            }

            lexicons = []
            for file in os.listdir(self.model_dir):
                file_path = os.path.join(self.model_dir, file)
                if file.endswith(".onnx"):
                    model_files["model"] = file_path
                elif file == "voices.bin":
                    model_files["kokoro_voices"] = file_path
                elif file == "lexicon.txt" or file == "lexicon-us-en.txt" or file == "lexicon-zh.txt":
                    lexicons.append(file_path)
                elif file == "tokens.txt":
                    model_files["tokens"] = file_path
                elif os.path.isdir(file_path) and file == "espeak-ng-data":
                    model_files["data_dir"] = file_path
                elif os.path.isdir(file_path) and file == "dict":
                    model_files["dict_dir"] = file_path
                elif file.endswith(".fst"):
                    model_files["rule_fsts"].append(file_path)

            if not model_files["model"]:
                raise FileNotFoundError("未找到ONNX模型文件")

            # 拼接多个lexicon路径
            model_files["lexicon"] = ",".join(lexicons)

            # 获取可选的 voices 字段，若没有则使用空字符串
            kokoro_voices = model_files.get("kokoro_voices", "")
            dict_dir = model_files.get("dict_dir", '')
            data_dir = model_files.get("data_dir", '')

            provider = detect_provider()
            # https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
            sid = 0
            num_threads = os.cpu_count()
            rule_fsts = ",".join(model_files["rule_fsts"]) if model_files["rule_fsts"] else ""

            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=model_files["model"],
                        lexicon=model_files["lexicon"],
                        dict_dir=model_files["dict_dir"] or '',
                        tokens=model_files["tokens"],
                        length_scale=self.speed,  # 设置语速
                    ),
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=model_files["model"],
                        voices=kokoro_voices,
                        tokens=model_files["tokens"],
                        lexicon=model_files["lexicon"],
                        data_dir=data_dir,
                        dict_dir=model_files["dict_dir"] or '',
                        length_scale=self.speed,
                    ),
                    provider=provider,
                    debug=False,
                    num_threads=num_threads,
                ),
                rule_fsts=rule_fsts,
                max_num_sentences=1,
            )

            if not tts_config.validate():
                raise ValueError("TTS 配置无效，请检查模型文件")

            tts = sherpa_onnx.OfflineTts(tts_config)

            global sample_rate, play_thread_started, started, stopped
            sample_rate = tts.sample_rate
            started = False
            stopped = False

            start = time.time()
            #Speech speed. Larger->faster; smaller->slower
            audio = tts.generate(text, sid=sid, speed=self.speed)
            end = time.time()
            logging.info(f"合成耗时: {end - start:.3f}秒")

            stopped = True

            if len(audio.samples) == 0:
                logging.info("生成失败，无音频")
                return

            elapsed_seconds = end - start
            audio_duration = len(audio.samples) / audio.sample_rate
            real_time_factor = elapsed_seconds / audio_duration
            
            State().pause_listening()  # 禁用监听
            
            # 播放音频
            sd.play(audio.samples, samplerate=tts.sample_rate, device=self.output_device)
            sd.wait()

            State().resume_listening()  # 启用监听

            logging.info(f"Audio duration: {audio_duration:.3f}s")
            logging.info(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")

        except Exception as e:
            logging.info(f"[ERROR] 合成失败: {e}")
            raise


if __name__ == "__main__":
    tts = TextToSpeech(
        backend="sherpa-onnx",
        model_dir="/path/to/sherpa-onnx/models",
        voice="zh-cn",
        speed=1.0,
    )

    try:
        tts.synthesize("你好，世界！", "output.wav")
    finally:
        stop_playback()
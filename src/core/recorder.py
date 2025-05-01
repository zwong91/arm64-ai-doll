import logging
import sounddevice as sd
import soundfile as sf

import numpy as np
import time
import noisereduce as nr
import sherpa_onnx

from ..utils.utils import resource_path

EXCLUDE_KEYWORDS = ["loopback", "mix", "stereo", "virtual", "monitor"]

def resolve_input_device(device):
    devices = sd.query_devices()

    # 如果是数字字符串，比如 "1"
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    # 如果是合法的设备编号
    if isinstance(device, int):
        if 0 <= device < len(devices) and devices[device]["max_input_channels"] > 0:
            return device, devices[device]["name"]
        else:
            logging.info(f"[WARN] 无效输入设备编号 {device}，尝试自动选择")

    # 如果是设备名称字符串
    if isinstance(device, str):
        for i, dev in enumerate(devices):
            if device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                return i, dev["name"]
        logging.info(f"[WARN] 找不到名为 '{device}' 的输入设备，尝试自动选择")

    # 自动 fallback 到第一个有输入通道的设备
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name = dev["name"].lower()
            if any(k in name for k in EXCLUDE_KEYWORDS):
                continue
            logging.info(f"[INFO] 自动选择输入设备: {dev['name']} (#{i})")
            return i, dev["name"]

    raise RuntimeError("未找到可用的输入设备")

class Recorder:
    def __init__(self, sample_rate=16000, input_device=None, vad_model_path="vad_ckpt/silero_vad.onnx"):
        self.sample_rate = sample_rate
        device_id, device_name = resolve_input_device("default")

        self.input_device = device_id
        self.device_name = device_name

        # 初始化VAD
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = resource_path(vad_model_path)

        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.25  # seconds
        vad_config.silero_vad.min_speech_duration = 0.25  # seconds
        # If the current segment is larger than this value, then it increases
        # the threshold to 0.9 internally. After detecting this segment,
        # it resets the threshold to its original value.
        vad_config.silero_vad.max_speech_duration = 5  # seconds

        vad_config.sample_rate = sample_rate
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

    @staticmethod
    def list_devices():
        devices = sd.query_devices()
        logging.info("\n可用的音频设备:")
        for i, dev in enumerate(devices):
            logging.info(f"{i}: {dev['name']} (输入通道: {dev['max_input_channels']}, 输出通道: {dev['max_output_channels']})")
        return devices

    def record_until_silence(self, silence_duration=1.0, enable_noise_reduction=True):
        chunk_duration = 0.1  # 秒
        chunk_size = int(self.sample_rate * chunk_duration)
        silence_chunks = int(silence_duration / chunk_duration)

        recorded = []
        silence_counter = 0
        speech_detected = False
        recording_done = False
        start_time = None

        logging.info("Microphone Listening for speech...")

        def callback(indata, frames, time_info, status):
            nonlocal recorded, silence_counter, speech_detected, start_time, recording_done
            if status:
                logging.info(status)

            chunk = indata[:, 0]
            self.vad.accept_waveform(chunk)

            if not speech_detected:
                if self.vad.is_speech_detected():
                    logging.info("Speech detected, start recording")
                    speech_detected = True
                    start_time = time.time()
                    recorded.append(chunk.copy())
            else:
                recorded.append(chunk.copy())
                if self.vad.is_speech_detected():
                    silence_counter = 0
                else:
                    silence_counter += 1

                if silence_counter >= silence_chunks:
                    logging.info("Silence detected, stop recording")
                    recording_done = True
                    raise sd.CallbackStop()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.input_device,
            blocksize=chunk_size,
            callback=callback,
        ):
            while not recording_done:
                time.sleep(0.05)

        # 获取 VAD 检测到的语音片段
        speech_samples = []
        while not self.vad.empty():
            speech_samples.extend(self.vad.front.samples)
            self.vad.pop()

        speech_samples = np.array(speech_samples, dtype=np.float32)
        # 对语音片段进行归一化和降噪
        if len(speech_samples) > 0:
            speech_samples = speech_samples / np.max(np.abs(speech_samples))  # 归一化

            if enable_noise_reduction:
                speech_samples = nr.reduce_noise(y=speech_samples, sr=self.sample_rate)
            filename_for_speech = time.strftime("%Y%m%d-%H%M%S-speech.wav")
            sf.write(filename_for_speech, speech_samples, samplerate=self.sample_rate)
            logging.info(f"语音片段已保存: {filename_for_speech}")

        return speech_samples

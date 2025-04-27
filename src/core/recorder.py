import sounddevice as sd
import numpy as np
import time
import noisereduce as nr
import sherpa_onnx

from ..utils.resource_utils import resource_path
class Recorder:
    def __init__(self, sample_rate=16000, input_device=None, vad_model_path="vad_ckpt/silero_vad.onnx"):
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.vad_model_path = resource_path(vad_model_path)
        if self.input_device is None:
            self.input_device = sd.default.device[0]
        # 初始化VAD
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path

        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.25  # seconds
        vad_config.silero_vad.min_speech_duration = 0.25  # seconds
        # If the current segment is larger than this value, then it increases
        # the threshold to 0.9 internally. After detecting this segment,
        # it resets the threshold to its original value.
        vad_config.silero_vad.max_speech_duration = 5  # seconds

        vad_config.sample_rate = sample_rate
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

    def record_until_silence(self, silence_duration=1.0, enable_noise_reduction=True):
        chunk_duration = 0.1  # 秒
        chunk_size = int(self.sample_rate * chunk_duration)
        silence_chunks = int(silence_duration / chunk_duration * 1.1)

        recorded = []
        silence_counter = 0
        speech_detected = False
        recording_done = False
        start_time = None

        print("Listening for speech...")

        def callback(indata, frames, time_info, status):
            nonlocal recorded, silence_counter, speech_detected, start_time, recording_done
            if status:
                print(status)

            chunk = indata[:, 0]
            self.vad.accept_waveform(chunk)

            if not speech_detected:
                if self.vad.is_speech_detected():
                    print("Speech detected, start recording")
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
                    print("Silence detected, stop recording")
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

        if recorded:
            all_audio = np.concatenate(recorded)
            all_audio = all_audio / np.max(np.abs(all_audio))  # 归一化
            if enable_noise_reduction:
                all_audio = nr.reduce_noise(y=all_audio, sr=self.sample_rate)
        else:
            all_audio = np.zeros(0)

        return all_audio

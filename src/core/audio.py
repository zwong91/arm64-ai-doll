import sounddevice as sd
import numpy as np
import soundfile as sf
import time
import sherpa_onnx

class AudioManager:
    def __init__(self, input_device=None, output_device=None, sample_rate=16000):
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate

    @staticmethod
    def list_devices():
        devices = sd.query_devices()
        print("\n可用的音频设备:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (输入通道: {dev['max_input_channels']}, 输出通道: {dev['max_output_channels']})")
        return devices

    def record(self, duration):
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            device=self.input_device,
            dtype=np.float32
        )
        sd.wait()
        return recording

    def record_until_silence(self, vad_model_path, max_duration=10, silence_duration=2.0):
        sample_rate = self.sample_rate
        chunk_duration = 0.5
        chunk_size = int(sample_rate * chunk_duration)
        silence_chunks = int(silence_duration / chunk_duration)

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path
        vad_config.sample_rate = sample_rate
        vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

        recorded = []
        silence_counter = 0
        speaking = False
        start_time = time.time()

        print("Listening...")

        while time.time() - start_time < max_duration:
            chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1,
                        device=self.input_device, dtype=np.float32)
            sd.wait()
            chunk = chunk.flatten()

            vad.accept_waveform(chunk)
            is_speech = vad.is_speech_detected()

            if is_speech:
                if not speaking:
                    print("Detected speech")
                speaking = True
                silence_counter = 0
                recorded.append(chunk)
            elif speaking:
                silence_counter += 1
                recorded.append(chunk)
                if silence_counter >= silence_chunks:
                    print("Detected silence, committing recording")
                    break

        return np.concatenate(recorded) if recorded else np.zeros(0)


    def play(self, filename):
        data, sr = sf.read(filename)
        sd.play(data, sr, device=self.output_device)
        sd.wait()

    def save(self, data, filename, sample_rate):
        sf.write(filename, data, sample_rate)

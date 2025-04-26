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
        self._playing = False

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

    def record_until_silence(self, vad_model_path, max_duration=10, silence_duration=1.5):
        sample_rate = self.sample_rate
        chunk_duration = 0.1
        chunk_size = int(sample_rate * chunk_duration)
        silence_chunks = int(silence_duration / chunk_duration)

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path
        vad_config.sample_rate = sample_rate
        vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

        print("Listening for speech...")

        # Step 1: 等待有人开始说话
        while True:
            chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1,
                        device=self.input_device, dtype=np.float32)
            sd.wait()
            chunk = chunk.flatten()
            vad.accept_waveform(chunk)
            if vad.is_speech_detected():
                print("Speech detected, start recording")
                break

        # Step 2: 开始录音直到静音结束
        recorded = [chunk]
        silence_counter = 0
        start_time = time.time()

        while time.time() - start_time < max_duration:
            chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1,
                        device=self.input_device, dtype=np.float32)
            sd.wait()
            chunk = chunk.flatten()
            vad.accept_waveform(chunk)
            recorded.append(chunk)

            if vad.is_speech_detected():
                silence_counter = 0
            else:
                silence_counter += 1
                if silence_counter >= silence_chunks:
                    print("Silence detected, stop recording")
                    break

        all_audio = np.concatenate(recorded) if recorded else np.zeros(0)
        
        if silence_counter > 0:
            end_index = -silence_counter * chunk_size
            all_audio = all_audio[:end_index] if end_index != 0 else all_audio

        return all_audio


    def play(self, filename):
        data, sr = sf.read(filename)
        self._playing = True
        sd.play(data, sr, device=self.output_device)

    def stop(self):
        self._playing = False
        sd.stop()

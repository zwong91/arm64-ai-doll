import sounddevice as sd
import numpy as np
import soundfile as sf
import wave

class AudioManager:
    def __init__(self, input_device=None, output_device=None, sample_rate=16000):
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        
    @staticmethod
    def list_devices():
        """列出所有可用的音频设备"""
        devices = sd.query_devices()
        print("\n可用的音频设备:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (输入通道: {dev['max_input_channels']}, "
                  f"输出通道: {dev['max_output_channels']})")
        return devices

    def record(self, duration):
        """使用指定输入设备录音"""
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            device=self.input_device,
            dtype=np.float32
        )
        sd.wait()
        return recording

    def play(self, filename):
        """使用指定输出设备播放音频"""
        data, sr = sf.read(filename)
        sd.play(data, sr, device=self.output_device)
        sd.wait()

    def save(self, data, filename, sample_rate):
        """保存音频数据到文件"""
        sf.write(filename, data, sample_rate)

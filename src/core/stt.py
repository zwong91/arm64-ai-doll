import sys
import os
from faster_whisper import WhisperModel

def resource_path(path: str) -> str:
    """
    返回运行时可以访问到的绝对路径：
    1) 如果用户传入的是绝对路径，就直接返回；
    2) 否则在打包后，从 sys._MEIPASS 里找（PyInstaller onefile）；
    3) 平时开发环境，就从当前工作目录找（os.path.abspath(".")）。
    """
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(path):
        return path

    # 打包运行时，PyInstaller 会把所有资源解压到这里
    base_path = getattr(sys, "_MEIPASS", None) or os.path.abspath(".")
    return os.path.join(base_path, path)

class SpeechToText:
    def __init__(self, model_path="whisper_ckpt", device="cpu", compute_type="int8"):
        # 示例模型 tiny，可以改成 base、small、medium、large-v3
        #model_size = "tiny"
        #self.model = WhisperModel(model_size, download_root="whisper_ckpt")

        real_path = resource_path(model_path)
        self.model = WhisperModel(real_path, device=device, compute_type=compute_type)

    def transcribe(self, audio_file):
        """将音频转换为文本"""
        segments, _ = self.model.transcribe(audio_file)
        return " ".join([segment.text for segment in segments])
import os
import sys
import subprocess
import numpy as np

import sherpa_onnx
import soundfile as sf

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


class TextToSpeech:
    def __init__(self, 
                 model_dir="vits-icefall-zh-aishell3",  # Path to the Sherpa-ONNX TTS model directory
                 backend="sherpa-onnx",
                 voice="zh-cn",   # Language/voice code
                 speed=0.8,       # Speaking rate (1.0 is normal speed)
        ):

        self.backend = backend
        self.voice = voice
        self.speed = speed
        
        real_path = resource_path(model_dir)
        # Path to the model directory is required for Sherpa-ONNX
        if real_path is None:
            raise ValueError("model_dir must be specified for Sherpa-ONNX backend")
        self.model_dir = real_path

        # Validate model directory exists
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def synthesize(self, text, output_file):
        """Convert text to speech using Sherpa-ONNX"""
        #print(f"[DEBUG] Using voice: {self.voice}")
        #print(f"[DEBUG] Using model directory: {self.model_dir}")
        if self.backend == "sherpa-onnx":
            self._synthesize_sherpa_onnx(text, output_file)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _synthesize_sherpa_onnx(self, text, output_file):
        """使用 Sherpa-ONNX API 生成语音"""
        import torch
        import platform
        import time

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

            for file in os.listdir(self.model_dir):
                file_path = os.path.join(self.model_dir, file)
                if file.endswith(".onnx"):
                    model_files["model"] = file_path
                elif file == "lexicon.txt":
                    model_files["lexicon"] = file_path
                elif file == "tokens.txt":
                    model_files["tokens"] = file_path
                elif os.path.isdir(file_path) and file == "dict":
                    model_files["dict_dir"] = file_path
                elif file.endswith(".fst"):
                    model_files["rule_fsts"].append(file_path)

            if not model_files["model"]:
                raise FileNotFoundError("未找到ONNX模型文件")

            provider = detect_provider()
            sid = 103
            num_threads = os.cpu_count()
            rule_fsts = ",".join(model_files["rule_fsts"]) if model_files["rule_fsts"] else ""

            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=model_files["model"],
                        lexicon=model_files["lexicon"],
                        #data_dir=self.model_dir,
                        dict_dir=model_files["dict_dir"] or '',
                        tokens=model_files["tokens"],
                        length_scale=self.speed,  # 设置语速
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

            # TODO: tts playback
            start = time.time()
            audio = tts.generate(text, sid=sid, speed=self.speed)
            end = time.time()

            if len(audio.samples) == 0:
                print("生成失败，无音频")
                return

            elapsed_seconds = end - start
            audio_duration = len(audio.samples) / audio.sample_rate
            real_time_factor = elapsed_seconds / audio_duration

            sf.write(
                output_file,
                audio.samples,
                samplerate=audio.sample_rate,
                subtype="PCM_16",
            )

            #print(f"Saved to {output_file}")
            #print(f"Text: '{text}'")
            #print(f"Elapsed: {elapsed_seconds:.3f}s")
            print(f"Audio duration: {audio_duration:.3f}s")
            print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")

        except Exception as e:
            print(f"[ERROR] 合成失败: {e}")
            raise


# Example usage:
if __name__ == "__main__":
    # Initialize the TTS with Sherpa-ONNX
    tts = TextToSpeech(
        backend="sherpa-onnx",
        model_dir="/path/to/sherpa-onnx/models",
        voice="zh-cn",   # Chinese language
        speed=1.0,       # Normal speed
        volume=1.0       # Normal volume
    )
    
    # Generate speech
    tts.synthesize("你好，世界！", "output.wav")
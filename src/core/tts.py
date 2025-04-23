import os
import sys
import subprocess
import numpy as np

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
                 speed=1.2,       # Speaking rate (1.0 is normal speed)
        ):

        self.backend = backend
        self.voice = voice
        self.speed = speed
        self.noise_scale = 0.5,
        self.noise_scale_w = 0.6,
        
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
        print(f"[DEBUG] Using voice: {self.voice}")
        print(f"[DEBUG] Using model directory: {self.model_dir}")
        
        if self.backend == "sherpa-onnx":
            self._synthesize_sherpa_onnx(text, output_file)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _synthesize_sherpa_onnx(self, text, output_file):
        """Generate speech using Sherpa-ONNX"""
        try:
            # Path to the Sherpa-ONNX TTS binary
            sherpa_onnx_bin = "./sherpa-onnx-offline-tts"
            #os.chmod(sherpa_onnx_bin, 0o755)  # 确保可执行           
            # Find model files in the model directory
            model_files = {
                "model": None,
                "lexicon": None,
                "tokens": None,
                "dict_dir": None,
                "rule_fsts": [],  # 存储规则文件
            }
            
            # Look for model files with common names
            for file in os.listdir(self.model_dir):
                file_path = os.path.join(self.model_dir, file)
                if file.endswith(".onnx"):
                    model_files["model"] = file_path
                elif file == "lexicon.txt":
                    model_files["lexicon"] = file_path
                elif file == "tokens.txt":
                    model_files["tokens"] = file_path
                elif os.path.isdir(file_path) and file == "dict":  # 查找 dict 子目录
                    model_files["dict_dir"] = file_path
                elif file.endswith(".fst"):  # 查找 .fst 文件
                    model_files["rule_fsts"].append(file_path)
            
            # Check if required files are found
            if not model_files["model"]:
                raise FileNotFoundError(f"No ONNX model file found in {self.model_dir}")
            
            # 设置sid和其他TTS相关的参数
            sid = 66  # 说话人ID
            
            # 动态生成 rule_fsts 参数
            rule_fsts = ",".join(model_files["rule_fsts"]) if model_files["rule_fsts"] else ""

            # Build the command with appropriate arguments for Sherpa-ONNX
            cmd = [
                sherpa_onnx_bin,
                f"--vits-model={model_files['model']}",
                f"--vits-lexicon={model_files['lexicon']}",
                f"--vits-tokens={model_files['tokens']}",
                f"--tts-rule-fsts={rule_fsts}",  # 添加规则文件参数
                f"--vits-length-scale={self.speed}",  # 可变语速
                f"--sid={sid}",
                f"--output-filename={output_file}",
                f"{text}",
            ] 
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"[INFO] Sherpa-ONNX TTS completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Sherpa-ONNX TTS failed: {e}")
            print(f"[ERROR] stdout: {e.stdout}")
            print(f"[ERROR] stderr: {e.stderr}")
            raise
            
        except Exception as e:
            print(f"[ERROR] Failed to generate speech: {e}")
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
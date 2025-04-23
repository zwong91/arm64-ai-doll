from dataclasses import dataclass

@dataclass
class Config:
    # STT配置
    whisper_model: str = "whisper_ckpt"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    
    # TTS配置
    tts_voice: str = "vits-icefall-zh-aishell3"
    
    # LLM配置
    llm_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "qwen2.5:0.5b"
    
    # 音频配置
    sample_rate: int = 16000
    record_duration: int = 5

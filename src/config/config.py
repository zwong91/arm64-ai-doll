from dataclasses import dataclass

@dataclass
class Config:
    # STT配置
    asr_model: str = "whisper"
    
    # TTS配置
    tts_voice: str = "vits-icefall-zh-aishell3"
    
    # LLM配置
    llm_model: str = "MiniMind2-Small"
    
    # 音频配置
    sample_rate: int = 16000
    record_duration: int = 5

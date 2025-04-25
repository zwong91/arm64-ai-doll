class Config:
    def __init__(self, 
                 asr_model='whisper',
                 tts_voice='vits-icefall-zh-aishell3',
                 llm_model='MiniMind2-Small',
                 input_device=None,  # 音频输入设备
                 output_device=None, # 音频输出设备
                 record_duration=5,
                 sample_rate=16000):
        self.asr_model = asr_model
        self.tts_voice = tts_voice
        self.llm_model = llm_model
        self.input_device = input_device
        self.output_device = output_device
        self.record_duration = record_duration
        self.sample_rate = sample_rate

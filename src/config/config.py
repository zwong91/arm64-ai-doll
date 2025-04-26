class Config:
    def __init__(self, 
                 asr_model='sensevoice',
                 input_device=None,  # 音频输入设备
                 output_device=None, # 音频输出设备
                 vad_model="silero_vad.onnx",  # VAD 模型路径
                 tts_voice='kokoro-multi-lang-v1_0',
                 llm_model='MiniMind2-Small',
                 record_duration=5,
                 sample_rate=16000):
        self.asr_model = asr_model
        self.denoiser_model = "speech-enhancement/gtcrn_simple.onnx"
        self.vad_model = vad_model
        self.input_device = input_device
        self.output_device = output_device
        self.tts_voice = tts_voice
        self.llm_model = llm_model
        self.record_duration = record_duration
        self.silence_duration = 2.0
        self.sample_rate = sample_rate

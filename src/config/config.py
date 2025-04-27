class Config:
    def __init__(
        self,
        asr_model: str = "whisper",
        input_device: str = "default",
        output_device: str = "default",
        vad_model: str = "vad_ckpt/silero_vad.onnx",
        sample_rate: int = 16000,
        tts_model: str = "sherpa/kokoro-multi-lang-v1_0",
        llm_model: str = "MiniMind2-Small",
        denoiser_model: str = "speech-enhancement/gtcrn_simple.onnx"
    ):
        self.asr_model = asr_model
        self.input_device = input_device
        self.output_device = output_device
        self.vad_model = vad_model
        self.sample_rate = sample_rate
        self.tts_model = tts_model
        self.llm_model = llm_model 
        self.denoiser_model = denoiser_model

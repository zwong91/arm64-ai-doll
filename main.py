import argparse
import tempfile
import os
import time
import numpy as np
import soundfile as sf

from src.core.stt import SpeechToText
from src.core.tts import TextToSpeech
from src.core.llm import LocalLLMClient
from src.core.audio import AudioManager
from src.config.config import Config
import sherpa_onnx

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

class VoiceAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.stt = SpeechToText(config.asr_model)
        self.tts = TextToSpeech(config.tts_voice)
        self.llm = LocalLLMClient(config.llm_model)
        self.audio = AudioManager(
            input_device=config.input_device,
            output_device=config.output_device,
            sample_rate=config.sample_rate
        )

    def _apply_vad(self, samples):
        sample_rate = self.config.sample_rate
        #samples_per_read = int(0.1 * sample_rate)

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = self.config.vad_model
        vad_config.sample_rate = sample_rate
        window_size = vad_config.silero_vad.window_size

        vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

        buffer = np.array([], dtype=np.float32)
        all_samples = samples
        buffer = np.concatenate([buffer, samples[:, 0]])

        while len(buffer) > window_size:
            vad.accept_waveform(buffer[:window_size])
            buffer = buffer[window_size:]

        speech_samples = []
        while not vad.empty():
            speech_samples.extend(vad.front.samples)
            vad.pop()

        return np.array(speech_samples, dtype=np.float32), all_samples

    def process_conversation(self):
        audio = self.audio.record_until_silence(self.config.vad_model)

        if audio is None or len(audio) == 0:
            print("未检测到语音")
            return

        print(f"录音长度: {len(audio) / self.config.sample_rate:.2f} 秒")
        print(f"最大音量: {np.max(np.abs(audio)):.4f}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            self.audio.save(audio, temp_input.name, self.config.sample_rate)
            text = self.stt.transcribe(temp_input.name)
            response = self.llm.get_response(text)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                self.tts.synthesize(response, temp_output.name)
                self.audio.play(temp_output.name)
                os.unlink(temp_output.name)

            os.unlink(temp_input.name)

    def process_audio_file(self, audio_file_path, output_dir="."):
        start_time = time.time()

        text = self.stt.transcribe(audio_file_path)
        print(f"语音识别耗时: {time.time() - start_time:.2f}秒")
        print(f"识别结果: {text}")

        response = self.llm.get_response(text)
        print(f"LLM响应耗时: {time.time() - start_time:.2f}秒")

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "response.wav")
        self.tts.synthesize(response, output_file)
        print(f"语音合成耗时: {time.time() - start_time:.2f}秒")
        print(f"Response saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Voice Assistant')
    parser.add_argument('--asr-model', default='whisper')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--file', '-f')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--list-devices', '-l', action='store_true')
    parser.add_argument('--input-device')
    parser.add_argument('--output-device')
    parser.add_argument('--pid-file')
    parser.add_argument('--vad-model', default='vad_ckpt/silero_vad.onnx', help='Path to silero_vad.onnx')
    args = parser.parse_args()

    if args.list_devices:
        AudioManager.list_devices()
        return

    config = Config(
        asr_model=args.asr_model,
        input_device=args.input_device,
        output_device=args.output_device,
        vad_model=args.vad_model
    )
    assistant = VoiceAssistant(config)

    if args.pid_file:
        with open(args.pid_file, 'w') as f:
            f.write(str(os.getpid()))

    if args.file:
        assistant.process_audio_file(args.file, args.output_dir)
    elif args.interactive:
        try:
            while True:
                assistant.process_conversation()
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("Please specify either --file or --interactive mode")

if __name__ == "__main__":
    main()


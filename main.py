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
from src.core.speech_denoiser import SpeechEnhancer
from src.config.config import Config
import langid

class VoiceAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.speech_enhancer = SpeechEnhancer(config.denoiser_model)
        self.stt = SpeechToText(config.asr_model)
        self.tts = TextToSpeech(config.tts_voice)
        self.llm = LocalLLMClient(config.llm_model)
        self.audio = AudioManager(
            input_device=config.input_device,
            output_device=config.output_device,
            sample_rate=config.sample_rate
        )

    def process_conversation(self):
        audio = self.audio.record_until_silence(self.config.vad_model, self.config.record_duration)
        print("VAD完成:", time.strftime("%H:%M:%S"))
        all_start = time.time()
        if audio is None or len(audio) == 0:
            print("未检测到语音")
            return

        print(f"录音长度: {len(audio) / self.config.sample_rate:.2f} 秒")
        print(f"最大音量: {np.max(np.abs(audio)):.4f}")
        enhanced_audio = self.speech_enhancer.enhance(audio, self.config.sample_rate)
        print(f"增强音频长度: {len(enhanced_audio) / self.config.sample_rate:.2f} 秒")
        print(f"增强最大音量: {np.max(np.abs(enhanced_audio)):.4f}")
        start = time.time()
        enhanced_audio = np.asarray(enhanced_audio)
        text = self.stt.transcribe(self.config.sample_rate, enhanced_audio)    
        language = langid.classify(text)[0].strip().lower()
        if language in ('zh', 'en'):
            print(f"Language detected: {language}")
        else:
            print(f"Unsupported language: {language}")
            return

        print(f"语音识别耗时: {time.time() - start:.2f}秒")

        start = time.time()
        response = self.llm.get_response(text)
        print(f"LLM响应耗时: {time.time() - start:.2f}秒")

        start = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            self.tts.synthesize(response, temp_output.name)
            print(f"语音合成耗时: {time.time() - start:.2f}秒")

            # TODO: 可打断
            #self.audio.play(temp_output.name)
            os.unlink(temp_output.name)

        print(f"总耗时: {time.time() - all_start:.2f}秒")


    def process_audio_file(self, audio_file_path, output_dir="."):
        all_start = time.time()

        start = time.time()
        text = self.stt.transcribe(audio_file_path)
        print(f"语音识别耗时: {time.time() - start:.2f}秒")
        print(f"识别结果: {text}")

        start = time.time()
        response = self.llm.get_response(text)
        print(f"LLM响应耗时: {time.time() - start:.2f}秒")

        start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "response.wav")
        self.tts.synthesize(response, output_file)
        print(f"语音合成耗时: {time.time() - start:.2f}秒")

        print(f"Response saved to: {output_file}")
        print(f"总耗时: {time.time() - all_start:.2f}秒")


def main():
    parser = argparse.ArgumentParser(description='Voice Assistant')
    parser.add_argument('--asr-model', default='whisper')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--file', '-f')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--list-devices', '-l', action='store_true')
    parser.add_argument('--input-device', default='default')
    parser.add_argument('--output-device', default='default')
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


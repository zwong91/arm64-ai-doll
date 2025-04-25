import argparse
import tempfile
import os
import time
from src.core.stt import SpeechToText
from src.core.tts import TextToSpeech
from src.core.llm import LocalLLMClient
from src.core.audio import AudioManager
from src.config.config import Config

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

    def process_conversation(self):
        """处理实时对话"""
        audio = self.audio.record(self.config.record_duration)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            self.audio.save(audio, temp_input.name, self.config.sample_rate)
            text = self.stt.transcribe(temp_input.name)
            print(f"You said: {text}")

            response = self.llm.get_response(text)
            print(f"AI response: {response}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                self.tts.synthesize(response, temp_output.name)
                self.audio.play(temp_output.name)
                os.unlink(temp_output.name)
            os.unlink(temp_input.name)

    def process_audio_file(self, audio_file_path, output_dir='.'):
        """处理音频文件"""
        start_time = time.time()
        
        # STT耗时
        stt_start = time.time()
        text = self.stt.transcribe(audio_file_path)
        stt_time = time.time() - stt_start
        print(f"语音识别耗时: {stt_time:.2f}秒")
        print(f"识别结果: {text}")

        # LLM耗时
        llm_start = time.time()
        response = self.llm.get_response(text)
        llm_time = time.time() - llm_start
        print(f"LLM响应耗时: {llm_time:.2f}秒")
        #print(f"AI回复: {response}")

        # TTS耗时
        tts_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "response.wav")
        self.tts.synthesize(response, output_file)
        tts_time = time.time() - tts_start
        print(f"语音合成耗时: {tts_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"\n总计耗时: {total_time:.2f}秒")
        print(f"Response saved to: {output_file}")


class AudioFileHandler():
    def __init__(self, assistant: VoiceAssistant, watch_dir: str):
        self.assistant = assistant 
        self.watch_dir = watch_dir
        self.output_dir = os.path.join(watch_dir, "responses")
        os.makedirs(self.output_dir, exist_ok=True)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(AUDIO_EXTENSIONS):
            file_path = event.src_path
            file_name = os.path.basename(file_path).lower()
            
            # 跳过响应音频文件
            if file_name.startswith("response_"):
                return
                
            print(f"\n检测到新音频文件: {file_name}")
            try:
                start_time = time.time()
                
                # 1. STT
                stt_start = time.time()
                text = self.assistant.stt.transcribe(file_path)
                stt_time = time.time() - stt_start
                print(f"语音识别耗时: {stt_time:.2f}秒") 
                print(f"识别结果: {text}")
                
                # 2. LLM
                llm_start = time.time()
                response = self.assistant.llm.get_response(text)
                llm_time = time.time() - llm_start
                print(f"LLM响应耗时: {llm_time:.2f}秒")
                print(f"AI回复: {response}")
                
                # 3. TTS
                tts_start = time.time()
                response_file = os.path.join(
                    self.output_dir, 
                    f"response_{os.path.splitext(file_name)[0]}.wav"
                )
                self.assistant.tts.synthesize(response, response_file)
                tts_time = time.time() - tts_start
                print(f"语音合成耗时: {tts_time:.2f}秒")
                
                # 4. 播放响应
                print("正在播放响应...")
                self.assistant.audio.play(response_file)
                
                total_time = time.time() - start_time
                print(f"\n总计耗时: {total_time:.2f}秒")
                print(f"响应已保存至: {response_file}")
                
            except Exception as e:
                print(f"处理文件出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='Voice Assistant')
    parser.add_argument('--asr-model', default='whisper', help='ASR model to use (e.g., whisper, sensevoice, etc.)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--file', '-f', help='Path to input audio file')
    parser.add_argument('--output-dir', help='Directory to save response audio file', default='.')
    parser.add_argument('--list-devices', '-l', action='store_true', help='List available audio devices')
    parser.add_argument('--input-device', help='Input device name or ID')
    parser.add_argument('--output-device', help='Output device name or ID')
    parser.add_argument('--pid-file', help='Path to save the process PID')
    args = parser.parse_args()

    # 支持音频输入设备选择（蓝牙耳机或麦克风）。 支持音频输出设备选择（蓝牙耳机或扬声器）。 录音和播放时指定设备。
    if args.list_devices:
        AudioManager.list_devices()
        return

    start_time = time.time()
    config = Config(
        asr_model=args.asr_model,
        input_device=args.input_device,
        output_device=args.output_device
    )
    assistant = VoiceAssistant(config)

    loading_time = time.time() - start_time
    print(f"initial model loading time: {loading_time:.4f} seconds")

    # 记录PID到文件
    if args.pid_file:
        with open(args.pid_file, 'w') as pid_file:
            pid_file.write(str(os.getpid()))
        print(f"PID saved to: {args.pid_file}")

    if args.file:
        assistant.process_audio_file(args.file, args.output_dir)
    elif args.interactive:
        while True:
            try:
                assistant.process_conversation()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
    else:
        print("Please specify either --file or --interactive mode")

if __name__ == "__main__":
    main()

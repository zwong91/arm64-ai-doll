import argparse
import tempfile
import os
import time
from src.core.stt import SpeechToText
from src.core.tts import TextToSpeech
from src.core.llm import LocalLLMClient
from src.config.config import Config


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

class VoiceAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.stt = SpeechToText(config.whisper_model, config.whisper_device, config.whisper_compute_type)
        self.tts = TextToSpeech(config.tts_voice)
        self.llm = LocalLLMClient(config.llm_model)

    def process_conversation(self):
        """处理实时对话"""
        audio = self.audio.record(self.config.record_duration, self.config.sample_rate)
        
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

    def process_audio_file(self, audio_file_path):
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
        print(f"AI回复: {response}")

        # TTS耗时
        tts_start = time.time()
        output_dir = os.path.dirname(audio_file_path)
        output_file = os.path.join(output_dir, "response.wav")
        self.tts.synthesize(response, output_file)
        tts_time = time.time() - tts_start
        print(f"语音合成耗时: {tts_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"\n总计耗时: {total_time:.2f}秒")
        print(f"Response saved to: {output_file}")


class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, assistant: VoiceAssistant, watch_dir: str):
        self.assistant = assistant
        self.watch_dir = watch_dir

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(AUDIO_EXTENSIONS):
            file_path = event.src_path
            file_name = os.path.basename(file_path).lower()
            if file_name == "response.wav":
                print(f"跳过文件: {file_name}")
                return

            print(f"检测到新音频文件: {event.src_path}")
            try:
                self.assistant.process_audio_file(event.src_path)
            except Exception as e:
                print(f"处理文件出错: {e}")

def monitor_directory(path_to_watch, assistant: VoiceAssistant):
    print(f"开始监听目录: {path_to_watch}")
    event_handler = AudioFileHandler(assistant, path_to_watch)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("停止监听")
    observer.join()
    

def main():
    parser = argparse.ArgumentParser(description='Voice Assistant')
    parser.add_argument('--watch-dir', help='Path to watch for audio files')
    parser.add_argument('--file', '-f', help='Path to input audio file')
    args = parser.parse_args()

    start_time = time.time()
    config = Config()
    assistant = VoiceAssistant(config)

    loading_time = time.time() - start_time
    print(f"initial model loading time: {loading_time:.4f} seconds")

    if args.file:
        assistant.process_audio_file(args.file)
    elif args.watch_dir:
        try:
            monitor_directory(args.watch_dir, assistant)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("Please specify either --file or --watch-dir mode")

if __name__ == "__main__":
    main()

import argparse
import tempfile
import os
import time
import numpy as np
import soundfile as sf
import logging
from typing import Generator, Optional
from contextlib import contextmanager

from src.core.kws import KeywordSpotter
from src.core.stt import SpeechToText
from src.core.stream_microphone import AsrHandler
from src.core.tts import TextToSpeech
from src.core.llm import LocalLLMClient
from src.core.recorder import Recorder
from src.core.speech_denoiser import SpeechEnhancer
from src.config.config import Config
import langid

import asyncio
from queue import Queue

from src.utils.utils import smart_split

class VoiceAssistant:
    def __init__(self, config: Config):
        self._setup_logging()
        self._validate_config(config)
        self.config = config
        self.tts_queue = Queue()
        
        try:
            self.kws = KeywordSpotter(
                tokens_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
                encoder_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                decoder_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                joiner_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
                keywords_file="keywords/keywords.txt"
            )
            self.asr_handler = AsrHandler(model_path="sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en")
            self.speech_enhancer = SpeechEnhancer(config.denoiser_model)

            self.stt = SpeechToText(config.asr_model)
            self.tts = TextToSpeech(config.tts_model)
            self.llm = LocalLLMClient(config.llm_model)
            self.recorder = Recorder(
                sample_rate=config.sample_rate,
                input_device=config.input_device,
                vad_model_path=config.vad_model
            )
            self.is_awake_mode = True  # 初始唤醒模式
        except Exception as e:
            logging.error(f"初始化组件失败: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def _validate_config(config: Config) -> None:
        required_fields = ['sample_rate', 'input_device', 'vad_model']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"配置缺少必要字段: {field}")

    @contextmanager
    def _temp_audio_file(self, suffix: str = ".wav"):
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            yield temp_file.name
        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    async def _synthesize_worker(self):
        """异步语音合成worker"""
        while True:
            if not self.tts_queue.empty():
                text = self.tts_queue.get()
                if text == "#END":
                    break
                    
                with self._temp_audio_file() as temp_output:
                    # 在线程池中执行同步TTS
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, 
                        self.tts.synthesize,
                        text, 
                        temp_output
                    )
            await asyncio.sleep(0.1)

    async def process(self):
        """异步处理对话"""
        # 启动TTS worker
        tts_task = asyncio.create_task(self._synthesize_worker())
        
        gen = self.asr_handler.handle()
        for sentence in gen:
            logging.info(f"Q:\n{sentence}")
            reply_gen = self.llm.stream_chat(sentence)
            logging.info("A:")
            
            for reply in reply_gen:
                print(reply, end="")
                # 将文本片段加入TTS队列
                self.tts_queue.put(reply)
                yield reply
                
            self.tts_queue.put("#END")
            yield "#refresh"
            print()
            logging.info("==================")
            
            # 等待TTS完成当前句子
            await tts_task
            
        # 清理
        self.executor.shutdown()

    def process_conversation(self) -> Optional[str]:
        try:
            audio = self.recorder.record_until_silence(self.config.silence_duration)
            if not self._validate_audio(audio):
                return None

            if self.is_awake_mode:
                with self._time_it("关键字唤醒"):           
                    result = self.kws.process_audio(audio)
                    if result:
                        logging.info(f"检测到关键词: {result}")
                        self.is_awake_mode = False  # 切换到语音识别模式
                        self._synthesize_response("我在,我在。")
                        return None
                    else:
                        logging.info("未检测到关键词")
                        return None

            text = self._process_audio_to_text(audio)
            if not text:
                text = "我听不懂你说什么"

            stream = True
            if stream:
                buffer = ""
                for delta in self._generate_response(text, stream=True):
                    buffer += delta
                    sentences = smart_split(buffer)
                    # 只处理完整的句子，保留最后一段 incomplete 的
                    for sentence in sentences[:-1]:
                        self._synthesize_response(sentence)
                    buffer = sentences[-1] if sentences else buffer

                # 处理剩下的内容
                if buffer.strip():
                    self._synthesize_response(buffer)
            else:
                response = self._generate_response(text)
                self._synthesize_response(response)
            return response

        except Exception as e:
            logging.error(f"处理对话时出错: {str(e)}")
            return None

    def _validate_audio(self, audio: np.ndarray) -> bool:
        if audio is None or len(audio) == 0:
            logging.info("未检测到语音")
            return False
        
        logging.info(f"VAD语音结束: {time.strftime('%H:%M:%S')}")

        duration = len(audio) / self.config.sample_rate
        max_volume = np.max(np.abs(audio))
        logging.info(f"录音长度: {duration:.2f}秒, 最大音量: {max_volume:.4f}")
        return True

    def _process_audio_to_text(self, audio: np.ndarray) -> Optional[str]:
        try:
            text = self.stt.transcribe(self.config.sample_rate, audio)
            language = langid.classify(text)[0].strip().lower()        
            #if language not in ('zh', 'en'):
            if language != 'zh':
                logging.warning(f"不支持的语言: {language}")
                return None

            return text
        except Exception as e:
            logging.error(f"音频转文字失败: {str(e)}")
            return None

    def _generate_response(self, text: str, stream: bool = False):
        with self._time_it("LLM响应"):
            return self.llm.get_response(text, None, stream=stream)

    def _synthesize_response(self, response: str) -> None:
        with self._time_it("语音合成"):
            with self._temp_audio_file() as temp_file:
                self.tts.synthesize(response, temp_file)

    @contextmanager
    def _time_it(self, task_name: str):
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            logging.info(f"{task_name}耗时: {duration:.2f}秒")

    def process_audio_file(self, wave_filename, output_dir="."):
        try:
            all_start = time.time()

            audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # only use the first channel
            audio = np.ascontiguousarray(audio)
            with self._time_it("关键字唤醒"):           
                result = self.kws.process_audio(audio)
                if result:
                    print(f"检测到关键词: {result}")
                    self.is_awake_mode = False  # 切换到语音识别模式

            with self._time_it("语音转录"):
                text = self.stt.transcribe(sample_rate, audio)

            stream = True
            if stream:
                buffer = ""
                for delta in self._generate_response(text, stream=True):
                    buffer += delta
                    sentences = smart_split(buffer)
                    # 只处理完整的句子，保留最后一段 incomplete 的
                    for sentence in sentences[:-1]:
                        self._synthesize_response(sentence)
                    buffer = sentences[-1] if sentences else buffer

                # 处理剩下的内容
                if buffer.strip():
                    self._synthesize_response(buffer)
            else:
                response = self._generate_response(text)
                self._synthesize_response(response)

            # os.makedirs(output_dir, exist_ok=True)
            # output_file = os.path.join(output_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-speech.wav")
            # with self._time_it("语音合成"):
            #     self.tts.synthesize(response, output_file)

            # logging.info(f"saved to: {output_file}")
            logging.info(f"总耗时: {time.time() - all_start:.2f}秒")
            # Sleep
            time.sleep(10)
        except Exception as e:
            logging.error(f"处理音频文件时出错: {str(e)}")

def main():
    try:
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
            Recorder.list_devices()
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
            logging.info("启动交互模式...")
            try:
                while True:
                    assistant.process_conversation()
                    ##asyncio.run(assistant.process())
            except KeyboardInterrupt:
                logging.info("用户终止程序")
        else:
            logging.error("请指定 --file 或 --interactive 模式")

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()


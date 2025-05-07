import argparse
import tempfile
import os
import time
import numpy as np
import logging
from typing import Generator, Optional
from contextlib import contextmanager

from src.core.kws import KeywordSpotter
from src.core.stt import SpeechToText
from src.core.tts import TextToSpeech, stop_playback
from src.core.llm import LocalLLMClient
from src.core.recorder import Recorder
from src.core.speech_denoiser import SpeechEnhancer
from src.core.share_state import State

from src.config.config import Config

import langid

import soundfile as sf

import asyncio
from queue import Queue

from src.utils.utils import smart_split
from src.config.wake_keywords import keywords
import re


def clean_repeats(text):
    # 處理連續詞語（1~4字）重複超過兩次
    for n in range(4, 0, -1):  # 先處理長詞，避免誤判
        pattern = rf'((\S{{{n}}}))(\2){{2,}}'
        text = re.sub(pattern, r'\1\2', text)
    
    # 處理單字元重複超過兩次（標點、單個字）
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    return text

class VoiceAssistant:
    def __init__(self, config: Config):
        self._setup_logging()
        self._validate_config(config)
        self.config = config
        self.tts_queue = Queue()
        
        try:
            # self.kws = KeywordSpotter(
            #     tokens_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
            #     encoder_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            #     decoder_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            #     joiner_path="sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
            #     keywords_file="keywords/keywords.txt"
            # )
            #self.speech_enhancer = SpeechEnhancer(config.denoiser_model)

            self.stt = SpeechToText(config.asr_model)
            self.tts = TextToSpeech(config.tts_model, config.output_device)
            self.llm = LocalLLMClient(config.llm_model)
            self.recorder = Recorder(
                sample_rate=config.sample_rate,
                input_device=config.input_device,
                vad_model_path=config.vad_model
            )
            self.is_awake_mode = True  # 初始唤醒模式
            self.keywords = keywords
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

    def kws(self, text):
        for kw in self.keywords:
            if kw in text:
                return kw
        return None

    def process_conversation(self) -> Optional[str]:
        try:
            audio = self.recorder.record(self.config.silence_duration)
            if not self._validate_audio(audio) or not State.listening():
                logging.info("未检测到语音或静音")
                return None

            text = self._process_audio_to_text(audio)
            if not text:
                text = "我听不懂你说什么"
                return None

            if self.is_awake_mode:
                result = self._check_kws(text)
                if result:
                    logging.info(f"检测到关键词: {result}")
                    self.is_awake_mode = False  # 切换到语音识别模式
                    self._synthesize_response("我在,我在。")
                    return None
                else:
                    logging.info(f"未检测到关键词: raw text: {text}")
                    return None

            stream = True
            if stream:
                buffer = ""
                seg_idx = 1  # 句子序号从 1 开始

                for delta in self._generate_response(text, stream=True):
                    buffer += delta
                    sentences = smart_split(buffer)

                    # 只处理完整的句子，保留最后一段 incomplete 的
                    complete = sentences[:-1]
                    for sentence in complete:
                        logging.info(f"seg {seg_idx}: {sentence}\n")
                        self._synthesize_response(clean_repeats(sentence))
                        seg_idx += 1

                    # 保留最后一个不完整的片段
                    buffer = sentences[-1] if sentences else buffer

                # 处理剩下的残余内容
                if buffer.strip():
                    logging.info(f"seg {seg_idx}: {buffer}\n")
                    self._synthesize_response(clean_repeats(buffer))

            else:
                response = self._generate_response(text)
                sentences = smart_split(response)
                seg_idx = 1
                for sentence in sentences:
                    logging.info(f"seg {seg_idx}: {sentence}\n")
                    self._synthesize_response(sentence)
                    seg_idx += 1

            return None

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
                logging.warning(f"不支持的语言: {language}, text: {text}")
                return None

            return text
        except Exception as e:
            logging.error(f"音频转文字失败: {str(e)}")
            return None

    def _check_kws(self, text: str):
        with self._time_it("关键字唤醒"):
            return self.kws(text)

    def _generate_response(self, text: str, stream: bool = False):
        with self._time_it("LLM响应"):
            return self.llm.get_response(text, None, stream=stream)

    def _synthesize_response(self, response: str) -> None:
        with self._time_it("语音合成播放"):
                self.tts.synthesize(response)

    @contextmanager
    def _time_it(self, task_name: str):
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            logging.info(f"{task_name}耗时: {duration:.2f}秒")

    def process_audio_file(self, wave_filename):
        try:
            all_start = time.time()

            audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # only use the first channel
            audio = np.ascontiguousarray(audio)

            with self._time_it("语音转录"):
                text = self.stt.transcribe(sample_rate, audio)  
                result = self._check_kws(text)
                if result:
                    logging.info(f"检测到关键词: {result}")
                    self.is_awake_mode = False  # 切换到语音识别模式
                else:
                    logging.info(f"未检测到关键词: raw text: {text}")

            stream = True
            if stream:
                buffer = ""
                seg_idx = 1  # 句子序号从 1 开始

                for delta in self._generate_response(text, stream=True):
                    buffer += delta
                    sentences = smart_split(buffer)

                    # 只处理完整的句子，保留最后一段 incomplete 的
                    complete = sentences[:-1]
                    for sentence in complete:
                        logging.info(f"seg {seg_idx}: {sentence}\n")
                        self._synthesize_response(clean_repeats(sentence))
                        seg_idx += 1

                    # 保留最后一个不完整的片段
                    buffer = sentences[-1] if sentences else buffer

                # 处理剩下的残余内容
                if buffer.strip():
                    logging.info(f"seg {seg_idx}: {buffer}\n")
                    self._synthesize_response(clean_repeats(buffer))

            else:
                response = self._generate_response(text)
                sentences = smart_split(response)
                seg_idx = 1
                for sentence in sentences:
                    logging.info(f"seg {seg_idx}: {sentence}\n")
                    self._synthesize_response(clean_repeats(sentence))
                    seg_idx += 1

            logging.info(f"总耗时: {time.time() - all_start:.2f}秒")
        except Exception as e:
            logging.error(f"处理音频文件时出错: {str(e)}")

def main():
    try:
        parser = argparse.ArgumentParser(description='Voice Assistant')
        parser.add_argument('--asr-model', default='sensevoice')
        parser.add_argument('--interactive', '-i', action='store_true')
        parser.add_argument('--file', '-f')
        parser.add_argument('--list-devices', '-l', action='store_true')
        parser.add_argument('--input-device', default='default')
        parser.add_argument('--output-device', default=None)
        parser.add_argument('--pid-file')
        parser.add_argument('--vad-model', default='vad_ckpt/silero_vad.onnx')
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
            assistant.process_audio_file(args.file)
        elif args.interactive:
            logging.info("Interactive mode started...")
            try:
                while True:
                    assistant.process_conversation()
                    ##asyncio.run(assistant.process())
            except KeyboardInterrupt:
                logging.info("Exiting interactive mode...")
        else:
            logging.error("请指定 --file 或 --interactive 模式")

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()


import unittest
import numpy as np
from src.core.kws import KeywordSpotter

class TestKeywordSpotter(unittest.TestCase):
    def setUp(self):
        self.kws = KeywordSpotter(
            tokens_path="path/to/tokens.txt",
            encoder_path="path/to/encoder.onnx",
            decoder_path="path/to/decoder.onnx",
            joiner_path="path/to/joiner.onnx",
            keywords_file="path/to/keywords.txt"
        )

    def test_process_audio(self):
        # 创建测试用的音频数据
        test_audio = np.zeros(16000, dtype=np.float32)  # 1秒的静音
        result = self.kws.process_audio(test_audio)
        self.assertIsNone(result)  # 静音应该没有检测到关键词

if __name__ == '__main__':
    unittest.main()

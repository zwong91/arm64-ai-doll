关键词检测：https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html#sherpa-onnx-kws-zipformer-wenetspeech-3-3m-2024-01-01-chinese
本质是一个非常小的语音识别模型，这里用它来实现语音唤醒（一直监听音频流），类似小爱同学
支持自定义、复数个关键词且不需要重新训练。

```bash
# 嵌入文本转token
sherpa-onnx-cli text2token \
  --tokens sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
  --tokens-type ppinyin \
  keywords_raw.txt keywords.txt
```
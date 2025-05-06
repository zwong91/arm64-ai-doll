from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
import torch
import random
from typing import Generator, Optional, List, Dict, Union

from threading import Thread
from queue import Queue

from ..utils.utils import resource_path

import warnings
warnings.filterwarnings('ignore')

@dataclass
class LLMConfig:
    model_path: str = 'MiniMind2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_seq_len: int = 128
    max_new_tokens: int = 64
    temperature: float = 0.7
    repetition_penalty: float = 1.2
    top_p: float = 0.92
    
# -*- coding: utf-8 -*-
DEFAULT_SYSTEM_PROMPT = (
    "你是一个温柔、聪明、会讲故事的AI小伙伴，名字叫“小柱”。"
    "你专为儿童设计，说话亲切可爱，喜欢用简单、生动的语言和孩子们聊天、讲故事、解答他们的小问题。"
    "在讲故事时，你会用丰富的想象力、温暖的语气，以及一些有趣的细节让故事变得特别好听。"
    "你永远很有耐心，会用轻声细语安抚小朋友，让他们觉得安心和开心。"
)

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)

class LocalLLMClient:
    def __init__(self, config: Union[str, LLMConfig]):
        # 如果传入字符串，转换为 LLMConfig
        if isinstance(config, str):
            self.config = LLMConfig(model_path=config)
        else:
            self.config = config
        self.model, self.tokenizer = self._init_model()

    def _init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(resource_path(self.config.model_path))
        model = AutoModelForCausalLM.from_pretrained(
            resource_path(self.config.model_path), 
            trust_remote_code=True
        ).eval().to(self.config.device)
        print(f'Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
        return model, tokenizer

    def _prepare_input(self, prompt: str, messages: Optional[List[Dict]] = None) -> str:
        messages = [{"role": "assistant", "content": DEFAULT_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": prompt})
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-self.config.max_seq_len:]

    def generate_stream_response(self, prompt: str, messages):
        try:
            
            new_prompt = self._prepare_input(prompt, messages)
            print(f'👶: {prompt}')
            inputs = self.tokenizer(new_prompt, return_tensors="pt", truncation=True).to(self.config.device)

            queue = Queue()
            streamer = CustomStreamer(self.tokenizer, queue)
            def _generate():
                self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    repetition_penalty=self.config.repetition_penalty,
                    top_p=self.config.top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer
                )

            Thread(target=_generate).start()

            while True:
                text = queue.get()
                if text is None:
                    #"finish_reason": "stop"
                    break
                yield text

        except Exception as e:
            yield f"[ERROR] {str(e)}"
        

    def get_response(self, prompt: str, messages: Optional[List[Dict]] = None, stream: bool = False):
        random.seed(random.randint(0, 2048))
        if stream:
            def stream_generator():
                for chunk in self.generate_stream_response(prompt, messages):
                    yield chunk

            return stream_generator()
        else:        
            new_prompt = self._prepare_input(prompt, messages)
            print(f'👶: {prompt}')
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.no_grad():
                inputs = self.tokenizer(
                    new_prompt, 
                    return_tensors='pt', 
                    truncation=True
                ).to(self.config.device)
                print('🤖️: ', end='', flush=True)
                generated_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + self.config.max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    top_p=self.config.top_p,
                    temperature=self.config.temperature,
                    repetition_penalty=self.config.repetition_penalty,
                )
                
                #截去 prompt 部分
                answer = self.tokenizer.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                print('\n')
                return answer


from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import random
from typing import Generator, Optional, List, Dict, Union
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LLMConfig:
    model_path: str = 'model/minimind_tokenizer'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_seq_len: int = 200
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.85
    
DEFAULT_SYSTEM_PROMPT = "你是小柱子，一个温柔、聪明、会讲故事的AI小伙伴，专为儿童设计。"

class LocalLLMClient:
    def __init__(self, config: Union[str, LLMConfig]):
        # 如果传入字符串，转换为 LLMConfig
        if isinstance(config, str):
            self.config = LLMConfig(model_path=config)
        else:
            self.config = config
        self.model, self.tokenizer = self._init_model()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def _init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        ).eval().to(self.config.device)
        return model, tokenizer

    def _prepare_input(self, prompt: str, messages: Optional[List[Dict]] = None) -> str:
        if messages is None:
            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": prompt})
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def get_response(self, prompt: str, messages: Optional[List[Dict]] = None) -> str:
        random.seed(random.randint(0, 2048))
        input_text = self._prepare_input(prompt, messages)
        
        print(f'👶: {prompt}')
        print('🤖️: ', end='', flush=True)

        with torch.no_grad():
            inputs = self.tokenizer(
                input_text, 
                return_tensors='pt', 
                truncation=True
            ).to(self.config.device)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            print(response.strip())
            return response.strip()

    def stream_chat(self, prompt: str, messages: Optional[List[Dict]] = None) -> Generator:
        with self._lock:
            input_text = self._prepare_input(prompt, messages)
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True
            ).to(self.config.device)

            print(f'👶: {prompt}')
            
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True
            )

            self.executor.submit(
                self.model.generate,
                inputs.input_ids,
                max_length=self.config.max_seq_len + self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                streamer=streamer
            )

            for text in streamer:
                yield text


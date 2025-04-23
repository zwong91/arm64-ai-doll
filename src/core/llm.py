import torch
import sys
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import apply_lora, load_lora
from argparse import Namespace
from model.model_lora import *


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resource_path(path: str) -> str:
    """
    返回运行时可以访问到的绝对路径：
    1) 如果用户传入的是绝对路径，就直接返回；
    2) 否则在打包后，从 sys._MEIPASS 里找（PyInstaller onefile）；
    3) 平时开发环境，就从当前工作目录找（os.path.abspath(".")）。
    """
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(path):
        return path

    # 打包运行时，PyInstaller 会把所有资源解压到这里
    base_path = getattr(sys, "_MEIPASS", None) or os.path.abspath(".")
    return os.path.join(base_path, path)

class LocalLLMClient:
    def __init__(self, args):
        args = Namespace(
            load=1,
            use_moe=False,
            model_mode=2,
            dim=512,
            n_layers=8,
            max_seq_len=128,
            lora_name='None',
            out_dir='output',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_new_tokens=64
        )
        self.model, self.tokenizer = self._init_model(args)
        self.model_mode = args.model_mode
        self.max_seq_len = args.max_seq_len
        self.temperature = 1
        self.top_p = 0.85
        self.device ='cuda' if torch.cuda.is_available() else 'cpu',
        self.stream = False
        self.max_new_tokens = args.max_new_tokens if hasattr(args, 'max_new_tokens') else 128

    def _init_model(self, args):
        real_path = resource_path('model/minimind_tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(real_path)
        if args.load == 0:
            moe_path = '_moe' if args.use_moe else ''
            modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'} #4 RLAIF
            ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

            model = MiniMindLM(LMConfig(
                dim=args.dim,
                n_layers=args.n_layers,
                max_seq_len=args.max_seq_len,
                use_moe=args.use_moe
            ))

            state_dict = torch.load(ckp, map_location=args.device)
            model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

            if args.lora_name != 'None':
                apply_lora(model)
                load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
        else:
            transformers_model_path = resource_path('MiniMind2-Small')
            tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
            model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)

        print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
        return model.eval().to(args.device), tokenizer

    def get_response(self, prompt: str, messages=None) -> str:
        """使用本地模型生成文本（支持chat模板和流式输出）"""
        setup_seed(random.randint(0, 2048))

        if messages is None:
            messages = [{
                "role": "system",
                "content": "你是小柱子，一个温柔、聪明、会讲故事的AI小伙伴，专为儿童设计。。"
            }]
        messages.append({"role": "user", "content": prompt})

        # 构造 prompt
        if self.model_mode != 0:
            new_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            new_prompt = new_prompt[-self.max_seq_len:]  # 简化裁剪逻辑
        else:
            new_prompt = self.tokenizer.bos_token + prompt

        print(f'👶: {prompt}')
        answer = ""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            inputs = self.tokenizer(new_prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device=device, dtype=torch.long)

            # 处理 pad_token_id 可能为 None
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            # 判断是否支持流式
            if not hasattr(self.model, "generate"):
                raise NotImplementedError("当前模型不支持 generate 方法")

            # 生成
            print('🤖️: ', end='', flush=True)
            res_y = self.model.generate(
                input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_seq_len,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=pad_token_id
            )

            output_ids = res_y[0][input_ids.shape[1]:]
            answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(answer.strip())

        return answer.strip()



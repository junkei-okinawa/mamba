import argparse
import time
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

parser = argparse.ArgumentParser(description="Text Generation")
parser.add_argument("--model-name", type=str, default="clibrain/mamba-2.8b-instruct-openhermes")
parser.add_argument("--tokenizer", type=str, default="clibrain/mamba-2.8b-instruct-openhermes")
parser.add_argument("--chat-template", type=str, default=None)
parser.add_argument("--system-prompt", type=str, default=None)
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--eos-token", type=str, default=None)# "<|endoftext|>"
parser.add_argument("--max-length", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--split-str", type=str, default="<|assistant|>")
args = parser.parse_args()

dtype = torch.float16

print(f"Loading model {args.model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

if args.chat_template is not None:
    tokenizer.chat_template = AutoTokenizer.from_pretrained(args.chat_template).chat_template

if args.eos_token is not None:
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token

print(f"""
===
tokenizer.chat_template:
{tokenizer.chat_template}
===
tokenizer.eos_token:
{tokenizer.eos_token}
===
tokenizer.pad_token:
{tokenizer.pad_token}
===
""")

model = MambaLMHeadModel.from_pretrained(args.model_name, device=args.device, dtype=dtype)

model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

start = time.time()

torch.random.manual_seed(0)

def create_messages(system_prompt: Optional[str], prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        print("===== SYSTEM PROMPT =====\n")
        print(system_prompt)
        print("\n===== SYSTEM PROMPT =====")
        messages.append(dict(role="system", content=system_prompt))
    
    print("===== USER PROMPT =====\n")
    print(prompt)
    print("\n===== USER PROMPT =====")
    messages.append(dict(role="user", content=prompt))
    return messages

messages = create_messages(args.system_prompt, args.prompt)

input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
    ).to(device=args.device)

max_length = len(messages[-1]['content']) + args.max_length

out = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    temperature=args.temperature,
    top_k=args.topk,
    top_p=args.topp,
    repetition_penalty=args.repetition_penalty,
)

decoded = tokenizer.batch_decode(out)
print(f"Decoded: {decoded}")
assistant_message = (
    decoded[0].split(f"{args.split_str}\n")[-1].replace(tokenizer.eos_token, "")
)
print("===== RESULT =====\n")
print(assistant_message)
print("\n===== RESULT =====")
print(f"Prompt length: {len(messages[-1]['content'])}, generation length: {len(assistant_message) - len(messages[-1]['content'])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")

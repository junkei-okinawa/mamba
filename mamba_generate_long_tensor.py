import argparse
import time
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

parser = argparse.ArgumentParser(description="Text Generation")
parser.add_argument("--model-name", type=str, default="DeepMount00/Mamba-QA-ITA-790m")
parser.add_argument("--tokenizer", type=str, default="DeepMount00/Mamba-QA-ITA-790m")
parser.add_argument("--question", type=str, default=None)
parser.add_argument("--context", type=str, default=None)
parser.add_argument("--eos-token", type=str, default=None)# "<|endoftext|>"
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

dtype = torch.float16

print(f"Loading model {args.model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

if args.eos_token is not None:
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token

# print(f"""
# ===
# tokenizer.eos_token:
# {tokenizer.eos_token}
# ===
# tokenizer.pad_token:
# {tokenizer.pad_token}
# ===
# """)

model = MambaLMHeadModel.from_pretrained(args.model_name, device=args.device, dtype=dtype)

model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

start = time.time()

torch.random.manual_seed(0)
message = f"{args.question}\n\nQ: {args.context}\nA:"
input_ids = torch.LongTensor([tokenizer.encode(message)]).to(device=args.device)
out = model.generate(input_ids=input_ids, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(out)
print("decoded: ", decoded)
answer = decoded[0].replace(message, "").split("\n\n")[0].strip()
# print("answer: ", answer)
answer = answer.replace(tokenizer.eos_token, "")

print("===== RESULT =====\n")
print(answer)
print("\n===== RESULT =====")
print(f"Prompt length: {len(message)}, generation length: {len(answer)}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")

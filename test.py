import torch
from transformers import GPT2TokenizerFast

import engines

PROMPT = "ROMEO:\nBut, soft! what light through yonder window breaks?\nIt is the east, and Juliet is the sun."

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
prompt_ids = tokenizer.encode(PROMPT)                                     # list[int]
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")   # [1, T]

model = engines.load_prealloc_kv_model()

with torch.no_grad():
    logits = model(input_ids, use_cache=False).logits                    # [1, T, 50257]

print(f"prompt: {len(prompt_ids)} tokens, logits {tuple(logits.shape)}")
print(f"next token: {tokenizer.decode(logits[0, -1].argmax())!r}")

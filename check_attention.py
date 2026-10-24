import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from attention import MyAttention

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

input_ids = tokenizer.encode("The meaning of life is", return_tensors="pt").to(device)

# Reference: unpatched HF attention
with torch.no_grad():
    ref = model(input_ids, use_cache=False).logits   # [1, T, vocab_size]

# Swap in MyAttention on every block
for block in model.transformer.h:
    block.attn = MyAttention(block.attn)

with torch.no_grad():
    mine = model(input_ids, use_cache=False).logits

print(f"shapes:        ref {tuple(ref.shape)}  mine {tuple(mine.shape)}")
print(f"max abs diff:  {(ref - mine).abs().max().item():.3e}")
print(f"allclose:      {torch.allclose(ref, mine, atol=1e-4)}")
print(f"next token:    ref {tokenizer.decode(ref[0, -1].argmax())!r}  mine {tokenizer.decode(mine[0, -1].argmax())!r}")

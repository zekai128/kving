import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained GPT-2 (smallest version, 117M params)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

def generate(prompt: str, n_tokens: int = 50, temperature: float = 0.8) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(n_tokens):
            # Full forward pass over the entire sequence every step — no cache
            logits = model(input_ids).logits          # [1, T, vocab_size]
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0])

if __name__ == "__main__":
    prompt = "The meaning of life is"
    print(f"Prompt: {prompt}")
    print(f"Generated: {generate(prompt)}")

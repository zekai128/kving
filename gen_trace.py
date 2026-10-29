"""Generate a seeded request trace.

Prompts are real text cut from tinyShakespeare and tokenized with GPT-2's tokenizer.
The request contents depend only on --seed, so traces at different --rate values contain
the same requests (same ids, prompts, output lengths) with different arrival times.

    python3 gen_trace.py --workload mixed --n 200 --rate 2.0
    python3 gen_trace.py --workload mixed --n 200 --rate inf      # saturation: all arrive at t=0
    python3 gen_trace.py --workload shared_prefix --n 200 --rate 2.0
"""
import argparse
import json
import math
import os

import numpy as np
from transformers import GPT2TokenizerFast

MAX_CONTEXT = 1024  # GPT-2's position limit: prompt_len + output_len must fit


def lognormal_lengths(rng, n, median, sigma, lo, hi):
    x = rng.lognormal(mean=math.log(median), sigma=sigma, size=n)
    return np.clip(np.round(x), lo, hi).astype(int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["mixed", "shared_prefix"], default="mixed")
    p.add_argument("--n", type=int, default=200, help="number of requests")
    p.add_argument("--rate", type=float, default=1.0, help="requests/s (Poisson); inf = all at t=0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text", default="data/tinyshakespeare.txt")
    p.add_argument("--prefix-len", type=int, default=400, help="shared_prefix: shared prompt length")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10**9  # silence the >1024 warning; we only slice the ids
    corpus = tok.encode(open(args.text).read())

    rng = np.random.default_rng(args.seed)           # request contents
    arrival_rng = np.random.default_rng(args.seed + 1)  # arrival times only

    if args.workload == "mixed":
        prompt_lens = lognormal_lengths(rng, args.n, median=128, sigma=0.8, lo=16, hi=512)
    else:
        suffix_lens = rng.integers(16, 65, size=args.n)
        prompt_lens = args.prefix_len + suffix_lens
    output_lens = lognormal_lengths(rng, args.n, median=128, sigma=0.8, lo=16, hi=512)
    output_lens = np.minimum(output_lens, MAX_CONTEXT - prompt_lens)

    shared_prefix = corpus[:args.prefix_len]

    if math.isinf(args.rate):
        arrivals = np.zeros(args.n)
    else:
        gaps = arrival_rng.exponential(1.0 / args.rate, size=args.n)
        arrivals = np.cumsum(gaps) - gaps[0]  # first request arrives at t=0

    requests = []
    for i in range(args.n):
        if args.workload == "mixed":
            n_tok = int(prompt_lens[i])
            start = int(rng.integers(0, len(corpus) - n_tok))
            prompt = corpus[start:start + n_tok]
        else:
            n_tok = int(prompt_lens[i]) - args.prefix_len
            start = int(rng.integers(args.prefix_len, len(corpus) - n_tok))
            prompt = shared_prefix + corpus[start:start + n_tok]
        requests.append({
            "id": i,
            "prompt": prompt,
            "output_len": int(output_lens[i]),
            "arrival": round(float(arrivals[i]), 6),
        })

    rate_str = "inf" if math.isinf(args.rate) else f"{args.rate:g}"
    out = args.out or f"traces/{args.workload}_n{args.n}_seed{args.seed}_rate{rate_str}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    meta = {"workload": args.workload, "n": args.n, "rate": args.rate if not math.isinf(args.rate) else "inf",
            "seed": args.seed}
    with open(out, "w") as f:
        json.dump({"meta": meta, "requests": requests}, f)

    total_prompt = int(prompt_lens.sum())
    total_out = int(output_lens.sum())
    print(f"wrote {out}: {args.n} requests, prompt tokens {total_prompt} "
          f"(mean {total_prompt / args.n:.0f}), output tokens {total_out} (mean {total_out / args.n:.0f}), "
          f"span {arrivals[-1]:.1f}s")


if __name__ == "__main__":
    main()

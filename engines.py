"""Engine interface shared by every kving engine, plus a dummy engine for testing the harness.

The harness (bench.py) drives an engine with exactly three calls:
    engine.add_request(req)   # a request has arrived
    engine.step()             # one iteration -> StepResult
    engine.has_work()         # anything waiting or running?
Every engine version (no cache, serial cache, batching, paging, ...) implements these.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from attention import MyAttention
from serialpreallockvattention import SerialPreallocKVAttention

import torch


@dataclass
class Request:
    id: int
    prompt: list[int]       # prompt token ids
    output_len: int         # generate exactly this many tokens (greedy, ignore EOS)
    arrival: float          # seconds after the start of the run


@dataclass
class StepResult:
    tokens: list[tuple[int, int]]                       # (req_id, token) produced this step
    finished: list[int] = field(default_factory=list)   # req ids that completed this step
    admitted: list[int] = field(default_factory=list)   # req ids moved waiting -> running this step
    preempted: list[int] = field(default_factory=list)  # req ids moved running -> waiting this step
    num_prefill_tokens: int = 0     # prompt tokens processed this step
    num_decode_tokens: int = 0      # single-token decodes this step
    kv_tokens_used: int = 0         # tokens whose K/V are stored after this step
    kv_tokens_reserved: int = 0     # token capacity reserved for KV after this step


class Engine:
    """Base class. Subclasses implement the three methods below."""

    def add_request(self, req: Request) -> None:
        raise NotImplementedError

    def step(self) -> StepResult:
        """Run one iteration. Must not return until the step's tokens are known on the CPU."""
        raise NotImplementedError

    def has_work(self) -> bool:
        raise NotImplementedError


def load_model(name: str = "gpt2", device: str = "cuda"):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(name).to(device)
    model.eval()
    return model


class DummyEngine(Engine):
    """Fake serial engine for testing the harness: no model, emits deterministic tokens.

    Serves one request at a time. Each step sleeps to imitate GPU time.
    """

    def __init__(self, step_time: float = 0.002, prefill_time_per_token: float = 0.00002):
        self.step_time = step_time
        self.prefill_time_per_token = prefill_time_per_token
        self.waiting: deque[Request] = deque()
        self.current: Request | None = None
        self.generated = 0

    def add_request(self, req):
        self.waiting.append(req)

    def has_work(self):
        return self.current is not None or bool(self.waiting)

    def step(self):
        result = StepResult(tokens=[])
        if self.current is None:
            self.current = self.waiting.popleft()
            self.generated = 0
            result.admitted.append(self.current.id)
            result.num_prefill_tokens = len(self.current.prompt)
            time.sleep(self.step_time + self.prefill_time_per_token * len(self.current.prompt))
        else:
            result.num_decode_tokens = 1
            time.sleep(self.step_time)

        req = self.current
        token = (req.id * 7919 + self.generated) % 50257
        result.tokens.append((req.id, token))
        self.generated += 1
        result.kv_tokens_used = len(req.prompt) + self.generated
        result.kv_tokens_reserved = result.kv_tokens_used

        if self.generated == req.output_len:
            result.finished.append(req.id)
            self.current = None
        return result

class NoCacheEngine(Engine):
    """Step 0 baseline: real GPT-2, serial, no KV cache.

    Every step reruns the model on prompt + all generated tokens and keeps only the last
    position's prediction. Greedy decoding, so its outputs are the correctness reference.
    """

    def __init__(self):
        self.model = load_model()
        self.waiting: deque[Request] = deque()
        self.current: Request | None = None
        self.generated: list[int] = []

    def add_request(self, req):
        self.waiting.append(req)

    def has_work(self):
        return self.current is not None or bool(self.waiting)

    @torch.no_grad()
    def step(self):
        result = StepResult(tokens=[])
        if self.current is None:
            self.current = self.waiting.popleft()
            self.generated = []
            result.admitted.append(self.current.id)

        req = self.current
        input_ids = torch.tensor([req.prompt + self.generated], dtype=torch.long,
                                 device=self.model.device)              # [1, T]
        logits = self.model(input_ids, use_cache=False).logits          # [1, T, 50257]
        token = logits[0, -1].argmax().item()                           # waits for the GPU

        self.generated.append(token)
        result.tokens.append((req.id, token))
        # No cache: every step processes the whole sequence, so all of it counts as prefill.
        result.num_prefill_tokens = input_ids.shape[1]

        if len(self.generated) == req.output_len:
            result.finished.append(req.id)
            self.current = None
        return result

def load_prealloc_kv_model(name: str = "gpt2", device: str = "cuda"):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(name).to(device)
    model.eval()
    for block in model.transformer.h:
        block.attn = SerialPreallocKVAttention(block.attn)
    return model

class SerialPreallocKVCacheEngine(Engine):
    def __init__(self):
        self.model = load_prealloc_kv_model()
        self.waiting: deque[Request] = deque()
        self.current: Request | None = None
        self.generated: list[int] = []

    def add_request(self, req):
        self.waiting.append(req)

    def has_work(self):
        return self.current is not None or bool(self.waiting)

    @torch.no_grad()
    def step(self):
        result = StepResult(tokens=[])
        if self.current is None:
            self.current = self.waiting.popleft()
            self.generated = []
            result.admitted.append(self.current.id)

        req = self.current
        if not self.generated:
            prompt_ids = torch.tensor([req.prompt], dtype=torch.long,
                                 device=self.model.device)              # [1, T]
            logits = self.model(prompt_ids, prefill=True, use_cache=False).logits   # [1, T, 50257]
            token = logits[0, -1].argmax().item()                           # waits for the GPU

            self.generated.append(token)
            result.tokens.append((req.id, token))
            result.num_prefill_tokens = len(req.prompt)
        else:
            decode_id = torch.tensor([[self.generated[-1]]], dtype=torch.long,
                                      device=self.model.device)            # [1, 1]
            pos = len(req.prompt) + len(self.generated) - 1    # position of generated[-1] in the full sequence
            position_ids = torch.tensor([[pos]], dtype=torch.long, device=self.model.device)
            logits = self.model(decode_id, position_ids=position_ids, prefill=False, cache_pos=pos,
                                use_cache=False).logits                      # [1, 1, 50257]
            token = logits[0, -1].argmax().item()                           # waits for the GPU

            self.generated.append(token)
            result.tokens.append((req.id, token))
            result.num_decode_tokens = 1

        # positions with K/V in the cache: the prompt plus every generated token except the newest
        result.kv_tokens_used = len(req.prompt) + len(self.generated) - 1
        result.kv_tokens_reserved = 1024

        if len(self.generated) == req.output_len:
            result.finished.append(req.id)
            self.current = None
        return result


# name -> zero-argument constructor; add your engines here
ENGINES = {
    "dummy": DummyEngine,
    "no_cache": NoCacheEngine,
    "serial_kv": SerialPreallocKVCacheEngine,
}

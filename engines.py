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
from staticbatchpreallockvattention import StaticBatchPreallocKVAttention

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


def load_static_batch_model(max_batch: int, name: str = "gpt2", device: str = "cuda"):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(name).to(device)
    model.eval()
    for block in model.transformer.h:
        block.attn = StaticBatchPreallocKVAttention(block.attn, max_batch=max_batch)
    return model


class StaticBatchEngine(Engine):
    """Step 2: static batching. Take up to max_batch waiting requests, left-pad them into one
    batch, and run that batch until every request in it finishes. Only then admit the next batch.

    Rows whose request has finished keep running (their outputs are ignored) until the
    batch ends: that wasted work is what continuous batching (step 3) removes.
    """

    MAX_SEQ_LEN = 1024
    PAD_ID = 50256      # any valid token id; the mask stops anything attending to it

    def __init__(self, max_batch: int = 8):
        self.max_batch = max_batch
        self.model = load_static_batch_model(max_batch)
        self.device = self.model.device
        self.waiting: deque[Request] = deque()
        self.batch: list[Request] = []          # row i of the cache belongs to batch[i]
        self.generated: list[list[int]] = []    # generated[i]: tokens produced so far for batch[i]
        self.pad_len = None                     # [B] long, left-pad length of each row
        self.T = 0                              # padded prompt length = first decode column

    def add_request(self, req):
        self.waiting.append(req)

    def has_work(self):
        return bool(self.batch) or bool(self.waiting)

    def _live(self, i):
        return len(self.generated[i]) < self.batch[i].output_len

    @torch.no_grad()
    def step(self):
        result = StepResult(tokens=[])

        if not self.batch:
            # ---- admit a new batch and prefill it ----
            while self.waiting and len(self.batch) < self.max_batch:
                self.batch.append(self.waiting.popleft())
            self.generated = [[] for _ in self.batch]
            result.admitted = [r.id for r in self.batch]

            prompts = [r.prompt for r in self.batch]
            self.T = max(len(p) for p in prompts)
            assert self.T + max(r.output_len for r in self.batch) <= self.MAX_SEQ_LEN, "batch exceeds context"
            self.pad_len = torch.tensor([self.T - len(p) for p in prompts], dtype=torch.long, device=self.device)
            input_ids = torch.tensor([[self.PAD_ID] * (self.T - len(p)) + p for p in prompts],
                                     dtype=torch.long, device=self.device)                      # [B, T]
            position_ids = (torch.arange(self.T, device=self.device)[None, :]
                            - self.pad_len[:, None]).clamp(min=0)                               # [B, T]
            logits = self.model(input_ids, position_ids=position_ids, prefill=True,
                                pad_len=self.pad_len, use_cache=False).logits                   # [B, T, V]
            result.num_prefill_tokens = sum(len(p) for p in prompts)   # real tokens, padding not counted
        else:
            # ---- one decode step for every row (finished rows too; their output is ignored) ----
            g = len(self.generated[0])                   # every row has stepped the same number of times
            cache_pos = self.T + g - 1                   # column of the newest token, shared by all rows
            last = [gen[-1] for gen in self.generated]
            input_ids = torch.tensor(last, dtype=torch.long, device=self.device)[:, None]       # [B, 1]
            position_ids = (cache_pos - self.pad_len)[:, None]                                  # [B, 1]
            logits = self.model(input_ids, position_ids=position_ids, prefill=False,
                                cache_pos=cache_pos, pad_len=self.pad_len, use_cache=False).logits
            result.num_decode_tokens = sum(self._live(i) for i in range(len(self.batch)))

        next_tokens = logits[:, -1].argmax(-1).tolist()  # waits for the GPU
        for i, (req, token) in enumerate(zip(self.batch, next_tokens)):
            if not self._live(i):
                self.generated[i].append(token)          # keep lengths equal; not reported
                continue
            self.generated[i].append(token)
            result.tokens.append((req.id, token))
            if len(self.generated[i]) == req.output_len:
                result.finished.append(req.id)

        # KV actually used: real cached tokens of unfinished requests. Reserved: the whole buffer.
        result.kv_tokens_used = sum(len(r.prompt) + len(self.generated[i]) - 1
                                    for i, r in enumerate(self.batch) if self._live(i))
        result.kv_tokens_reserved = self.max_batch * self.MAX_SEQ_LEN

        if not any(self._live(i) for i in range(len(self.batch))):
            self.batch = []
        return result


# name -> zero-argument constructor; add your engines here
ENGINES = {
    "dummy": DummyEngine,
    "no_cache": NoCacheEngine,
    "serial_kv": SerialPreallocKVCacheEngine,
    "static_b8": StaticBatchEngine,
}

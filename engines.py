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


# name -> zero-argument constructor; add your engines here
ENGINES = {
    "dummy": DummyEngine,
}

"""Serve a trace through an engine and record raw per-request and per-step timings.

    python3 bench.py --engine dummy --trace traces/mixed_n200_seed0_rate2.json

Two processes, connected by a queue:
    client (load generator)  sends each trace request at its arrival time, then a None sentinel
    server (this process)    drains the queue into the engine, steps it, records timings

Writes results/<engine>__<trace name>.json. All metrics are computed later by analyze.py.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict, dataclass, field

from engines import ENGINES, Request

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@dataclass
class RequestRecord:
    id: int
    prompt_len: int
    output_len: int
    arrival: float               # when the client sent it; all times are seconds after the run start
    admit: float | None = None   # start of the step that admitted it
    first_token: float | None = None
    finish: float | None = None
    preemptions: int = 0
    tokens: list[int] = field(default_factory=list)


@dataclass
class StepRecord:
    t: float                     # end of step
    duration: float
    num_running: int             # distinct requests that produced a token
    num_prefill_tokens: int
    num_decode_tokens: int
    queue_len: int               # arrived but not yet admitted, after this step
    kv_tokens_used: int
    kv_tokens_reserved: int


def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def warmup(engine, trace_requests, n=3, output_len=8):
    """Run a few short requests to completion so one-time startup costs aren't timed."""
    for i, r in enumerate(trace_requests[:n]):
        engine.add_request(Request(id=-(i + 1), prompt=r["prompt"][:64], output_len=output_len, arrival=0.0))
    while engine.has_work():
        engine.step()
    sync()


def client(trace_requests, q):
    """Load generator. Runs in its own process and sends requests on schedule, whether or
    not the server is keeping up (open loop).

    Messages: the start time (float), then one dict per request, then None.
    """
    start = time.perf_counter()  # CLOCK_MONOTONIC: comparable across processes
    q.put(start)
    for r in sorted(trace_requests, key=lambda r: r["arrival"]):
        delay = start + r["arrival"] - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        q.put({**r, "sent": time.perf_counter() - start})
    q.put(None)


def run(engine, q):
    records = {}
    steps = []
    n_added = 0
    n_admitted = 0
    max_send_lag = 0.0   # how far the client fell behind its schedule
    client_done = False

    if HAS_CUDA:
        torch.cuda.reset_peak_memory_stats()
    start = q.get()  # blocks until the client process is up

    while not client_done or engine.has_work():
        # 1. Move every request in the queue to the engine. If the engine is idle, block
        #    until one arrives (the OS wakes us); otherwise just take what's there.
        block = not engine.has_work()
        while not client_done:
            try:
                msg = q.get(block=block)
            except queue.Empty:
                break
            block = False
            if msg is None:
                client_done = True
                break
            records[msg["id"]] = RequestRecord(id=msg["id"], prompt_len=len(msg["prompt"]),
                                               output_len=msg["output_len"], arrival=msg["sent"])
            engine.add_request(Request(id=msg["id"], prompt=msg["prompt"], output_len=msg["output_len"],
                                       arrival=msg["sent"]))
            max_send_lag = max(max_send_lag, msg["sent"] - msg["arrival"])
            n_added += 1

        if not engine.has_work():
            continue

        # 2. One engine step.
        t0 = time.perf_counter() - start
        result = engine.step()
        sync()
        t1 = time.perf_counter() - start

        for req_id in result.admitted:
            rec = records[req_id]
            if rec.admit is None:   # keep the first admission if a request is preempted and re-admitted
                rec.admit = t0      # admitted at the start of this step; its prefill ran during it
        n_admitted += len(result.admitted)
        for req_id in result.preempted:
            records[req_id].preemptions += 1
        n_admitted -= len(result.preempted)

        for req_id, token in result.tokens:
            rec = records[req_id]
            if rec.first_token is None:
                rec.first_token = t1
            rec.tokens.append(token)
        for req_id in result.finished:
            records[req_id].finish = t1

        steps.append(StepRecord(
            t=t1, duration=t1 - t0,
            num_running=len({req_id for req_id, _ in result.tokens}),
            num_prefill_tokens=result.num_prefill_tokens,
            num_decode_tokens=result.num_decode_tokens,
            queue_len=n_added - n_admitted,
            kv_tokens_used=result.kv_tokens_used,
            kv_tokens_reserved=result.kv_tokens_reserved,
        ))

    wall = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() if HAS_CUDA else 0
    return list(records.values()), steps, wall, peak_mem, max_send_lag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=sorted(ENGINES))
    p.add_argument("--trace", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    trace = json.load(open(args.trace))
    engine = ENGINES[args.engine]()

    if not args.no_warmup:
        warmup(engine, trace["requests"])

    # spawn, not fork: forking a process that has initialized CUDA is unsafe
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    client_proc = ctx.Process(target=client, args=(trace["requests"], q), daemon=True)
    client_proc.start()
    records, steps, wall, peak_mem, max_send_lag = run(engine, q)
    client_proc.join()

    unfinished = [r.id for r in records if r.finish is None]
    if unfinished:
        print(f"WARNING: {len(unfinished)} requests never finished: {unfinished[:10]}")

    trace_name = os.path.splitext(os.path.basename(args.trace))[0]
    out = args.out or f"results/{args.engine}__{trace_name}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "meta": {"engine": args.engine, "trace": args.trace, **trace["meta"],
                     "wall_time": wall, "peak_mem_bytes": peak_mem, "max_send_lag": max_send_lag},
            "requests": [asdict(r) for r in records],
            "steps": [asdict(s) for s in steps],
        }, f)

    n_tokens = sum(len(r.tokens) for r in records)
    print(f"wrote {out}: {len(records)} requests, {len(steps)} steps, {n_tokens} output tokens "
          f"in {wall:.1f}s ({n_tokens / wall:.0f} tok/s), client max send lag {1000 * max_send_lag:.1f}ms")


if __name__ == "__main__":
    main()

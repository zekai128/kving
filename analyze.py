"""Compute metrics from bench.py result files.

    python3 analyze.py results/*.json
    python3 analyze.py results/*.json --ref results/no_cache__mixed_n200_seed0_rateinf.json
    python3 analyze.py results/*.json --slo-ttft 2.0 --slo-tpot 0.05

Prints one row per run, then goodput per (engine, workload): the highest request rate whose
p99 TTFT and p99 TPOT both meet the SLOs. Runs at different rates of the same workload and
seed contain the same requests, so any run can serve as the correctness reference.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def summarize(run, ref_tokens=None):
    reqs = [r for r in run["requests"] if r["finish"] is not None]
    steps = run["steps"]

    ttft = [r["first_token"] - r["arrival"] for r in reqs]
    queue = [r["admit"] - r["arrival"] for r in reqs if r["admit"] is not None]
    prefill = [r["first_token"] - r["admit"] for r in reqs if r["admit"] is not None]
    tpot = [(r["finish"] - r["first_token"]) / (len(r["tokens"]) - 1) for r in reqs if len(r["tokens"]) > 1]
    e2e = [r["finish"] - r["arrival"] for r in reqs]

    n_out = sum(len(r["tokens"]) for r in reqs)
    span = max(r["finish"] for r in reqs) - min(r["arrival"] for r in reqs) if reqs else float("nan")

    decode_steps = [s for s in steps if s["num_prefill_tokens"] == 0]
    reserved = [s for s in steps if s["kv_tokens_reserved"] > 0]

    m = {
        "engine": run["meta"]["engine"],
        "workload": run["meta"]["workload"],
        "rate": run["meta"]["rate"],
        "n_done": len(reqs),
        "n_total": len(run["requests"]),
        "throughput": n_out / span,                      # output tokens/s
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "queue_p50": pct(queue, 50), "queue_p99": pct(queue, 99),
        "prefill_p50": pct(prefill, 50), "prefill_p99": pct(prefill, 99),
        "tpot_p50": pct(tpot, 50), "tpot_p99": pct(tpot, 99),
        "e2e_p50": pct(e2e, 50), "e2e_p99": pct(e2e, 99),
        "mean_batch": float(np.mean([s["num_running"] for s in steps])) if steps else 0.0,
        "decode_step_ms": 1000 * float(np.mean([s["duration"] for s in decode_steps])) if decode_steps else float("nan"),
        "max_queue": max((s["queue_len"] for s in steps), default=0),
        "kv_util": float(np.mean([s["kv_tokens_used"] / s["kv_tokens_reserved"] for s in reserved])) if reserved else float("nan"),
        "preemptions": sum(r["preemptions"] for r in run["requests"]),
        "peak_mem_gb": run["meta"]["peak_mem_bytes"] / 1e9,
    }

    if ref_tokens is not None:
        mismatched = [r["id"] for r in run["requests"] if r["tokens"] != ref_tokens.get(r["id"])]
        m["correct"] = "OK" if not mismatched else f"FAIL({len(mismatched)})"
    return m


def rate_key(rate):
    return float("inf") if rate == "inf" else float(rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--ref", default=None, help="result file whose outputs are the correctness reference")
    p.add_argument("--slo-ttft", type=float, default=2.0, help="p99 TTFT target, seconds")
    p.add_argument("--slo-tpot", type=float, default=0.05, help="p99 TPOT target, seconds")
    args = p.parse_args()

    ref = json.load(open(args.ref)) if args.ref else None
    ref_tokens = {r["id"]: r["tokens"] for r in ref["requests"]} if ref else None

    def same_requests(run):
        keys = ("workload", "n", "seed")
        return ref is not None and all(run["meta"][k] == ref["meta"][k] for k in keys)

    rows = []
    for path in args.results:
        run = json.load(open(path))
        m = summarize(run, ref_tokens if same_requests(run) else None)
        if ref is not None and "correct" not in m:
            m["correct"] = "-"      # different requests than the reference; not comparable
        rows.append(m)
    rows.sort(key=lambda m: (m["workload"], m["engine"], rate_key(m["rate"])))

    # (column, width, format spec)
    cols = [("engine", 14, "<"), ("workload", 14, "<"), ("rate", 6, ">"), ("n_done", 6, ">"),
            ("throughput", 10, ">.0f"), ("ttft_p50", 8, ">.3f"), ("ttft_p99", 8, ">.3f"),
            ("queue_p99", 9, ">.3f"), ("tpot_p50", 8, ">.4f"), ("tpot_p99", 8, ">.4f"),
            ("mean_batch", 10, ">.1f"), ("decode_step_ms", 14, ">.2f"), ("max_queue", 9, ">"),
            ("kv_util", 7, ">.2f"), ("preemptions", 11, ">"), ("peak_mem_gb", 11, ">.2f")]
    if ref_tokens is not None:
        cols.append(("correct", 10, ">"))

    print(" ".join(f"{name:{spec[0]}{width}}" for name, width, spec in cols))
    for m in rows:
        print(" ".join(f"{m[name]:{spec[0]}{width}{spec[1:]}}" for name, width, spec in cols))

    # Goodput: highest finite rate meeting both p99 SLOs, per (engine, workload).
    print(f"\ngoodput (p99 TTFT <= {args.slo_ttft}s, p99 TPOT <= {args.slo_tpot * 1000:.0f}ms):")
    groups = defaultdict(list)
    for m in rows:
        if rate_key(m["rate"]) != float("inf"):
            groups[(m["engine"], m["workload"])].append(m)
    for (engine, workload), ms in sorted(groups.items()):
        ok = [rate_key(m["rate"]) for m in ms
              if m["n_done"] == m["n_total"] and m["ttft_p99"] <= args.slo_ttft and m["tpot_p99"] <= args.slo_tpot]
        tested = ", ".join(f"{rate_key(m['rate']):g}" for m in ms)
        best = f"{max(ok):g} req/s" if ok else "none of the tested rates"
        print(f"  {engine:<14} {workload:<14} {best:<26} (tested: {tested})")


if __name__ == "__main__":
    main()

"""Re-aggregate raw per-request JSONL into summary metrics.

Used to recover results from raw data when summary.jsonl was not saved.
"""
import json
import os
import sys
import re
import numpy as np


def pct(data, p):
    if not data:
        return None
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def aggregate_file(filepath: str) -> dict:
    fname = os.path.basename(filepath)
    m = re.match(r"(.+?)_(short|long|reasoning)_c(\d+)_r(\d+)\.jsonl", fname)
    if not m:
        return None

    engine, regime, concurrency, repeat = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    concurrency = int(concurrency)

    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if not successful:
        return {"error": "all requests failed", "engine": engine, "regime": regime,
                "concurrency": concurrency, "repeat": repeat}

    output_tokens_list = [r["output_tokens"] for r in successful if r.get("output_tokens")]
    ttft_list = [r["ttft"] for r in successful if r.get("ttft") is not None]
    wall_list = [r["wall_time"] for r in successful]
    total_output_tokens = sum(output_tokens_list)

    wall_seconds = sum(wall_list) / concurrency if concurrency else sum(wall_list)
    throughput = total_output_tokens / wall_seconds if wall_seconds > 0 else 0

    tpot_list = []
    for r in successful:
        if r.get("ttft") and r.get("output_tokens", 0) > 1:
            tpot_list.append((r["wall_time"] - r["ttft"]) / (r["output_tokens"] - 1))

    reasoning_tok_list = [r.get("reasoning_tokens", 0) for r in successful]
    answer_tok_list = [r.get("answer_tokens", 0) for r in successful]
    ttft_answer_list = [r.get("ttft_answer") for r in successful if r.get("ttft_answer") is not None]

    return {
        "engine": engine,
        "regime": regime,
        "concurrency": concurrency,
        "repeat": repeat,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_seconds": round(wall_seconds, 3),
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_sec": round(throughput, 2),
        "throughput_per_request": round(throughput / concurrency, 2) if concurrency else 0,
        "ttft_p50": round(pct(ttft_list, 50), 4) if ttft_list else None,
        "ttft_p95": round(pct(ttft_list, 95), 4) if ttft_list else None,
        "tpot_p50": round(pct(tpot_list, 50), 4) if tpot_list else None,
        "tpot_p95": round(pct(tpot_list, 95), 4) if tpot_list else None,
        "latency_p50": round(pct(wall_list, 50), 4),
        "latency_p95": round(pct(wall_list, 95), 4),
        "latency_p99": round(pct(wall_list, 99), 4),
        "output_tokens_mean": round(float(np.mean(output_tokens_list)), 1) if output_tokens_list else 0,
        "reasoning_tokens_mean": round(float(np.mean(reasoning_tok_list)), 1) if reasoning_tok_list else 0,
        "answer_tokens_mean": round(float(np.mean(answer_tok_list)), 1) if answer_tok_list else 0,
        "ttft_answer_p50": round(pct(ttft_answer_list, 50), 4) if ttft_answer_list else None,
    }


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
    summary_path = os.path.join(os.path.dirname(__file__), "..", "results", "summary.jsonl")

    if not os.path.isdir(raw_dir):
        print(f"No raw data at {raw_dir}")
        sys.exit(1)

    summaries = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".jsonl"):
            continue
        result = aggregate_file(os.path.join(raw_dir, fname))
        if result:
            summaries.append(result)
            e, r, c, rep = result["engine"], result["regime"], result["concurrency"], result["repeat"]
            tp = result.get("throughput_tokens_per_sec", 0)
            ttft = result.get("ttft_p50", "N/A")
            lat = result.get("latency_p95", "N/A")
            print(f"  {e:<10} {r:<7} c={c:<3} r={rep}  tp={tp:.1f} tok/s  ttft_p50={ttft}  lat_p95={lat}")

    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")
    print(f"\nWrote {len(summaries)} entries → {summary_path}")

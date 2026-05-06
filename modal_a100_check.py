"""A100 validation run — vLLM FP16 + AWQ, c=1/16/64, short regime, 1 repeat.

Single container, two server starts. Total: 6 benchmark calls. ~$0.30-0.50.

Usage: modal run modal_a100_check.py
Results: results/a100_results.jsonl
"""
import modal
import os
import subprocess
import time
import json
from bench_lib import (
    MODEL_FP16, MODEL_AWQ,
    hf_cache, results_vol,
    make_vllm_image,
    VLLM_SERVER_ARGS, VLLM_AWQ_SERVER_ARGS,
    run_benchmark_impl, wait_for_server,
)

# Override GPU — import above already captured L4 for module-level constants
# but the class decorator below uses the string directly
GPU = "A100"
CONCURRENCIES = [1, 16, 64]

app = modal.App("inference-bench-a100")

@app.cls(
    image=make_vllm_image(),
    gpu=GPU,
    timeout=60 * 30,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class A100Validation:
    def _start_server(self, args):
        self.proc = subprocess.Popen(args)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), f"Server failed: {args[:4]}"
        return time.perf_counter() - t0

    def _stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    @modal.method()
    def run_fp16(self) -> list[dict]:
        t = self._start_server(VLLM_SERVER_ARGS)
        results = []
        for c in CONCURRENCIES:
            r = run_benchmark_impl("vllm", "short", c, repeat=1)
            r["cold_start_seconds"] = round(t, 1)
            r["config"] = "fp16"
            results.append(r)
        self._stop_server()
        return results

    @modal.method()
    def run_awq(self) -> list[dict]:
        t = self._start_server(VLLM_AWQ_SERVER_ARGS)
        results = []
        for c in CONCURRENCIES:
            r = run_benchmark_impl("vllm", "short", c, repeat=1, model_name=MODEL_AWQ)
            r["cold_start_seconds"] = round(t, 1)
            r["config"] = "awq"
            results.append(r)
        self._stop_server()
        return results


@app.local_entrypoint()
def main():
    print("=" * 50)
    print("  A100 validation: vLLM FP16 + AWQ")
    print(f"  Model: {MODEL_FP16} / {MODEL_AWQ}")
    print(f"  GPU: {GPU}")
    print(f"  Concurrencies: {CONCURRENCIES}")
    print("=" * 50)

    bench = A100Validation()

    print("\n--- FP16 ---")
    fp16 = bench.run_fp16.remote()
    for r in fp16:
        print(f"  c={r['concurrency']} | throughput={r.get('tokens_per_second', 0):.0f} tok/s | ttft={r.get('ttft_avg', 0):.2f}s")

    print("\n--- AWQ ---")
    awq = bench.run_awq.remote()
    for r in awq:
        print(f"  c={r['concurrency']} | throughput={r.get('tokens_per_second', 0):.0f} tok/s | ttft={r.get('ttft_avg', 0):.2f}s")

    all_results = fp16 + awq
    outpath = "results/a100_results.jsonl"
    with open(outpath, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\n  Done. {len(all_results)} results → {outpath}")

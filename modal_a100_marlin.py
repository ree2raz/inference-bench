"""A100 AWQ-Marlin test — runs c=1 and c=64 to check if Marlin kernel recovers efficiency.

Usage: modal run modal_a100_marlin.py
"""
import modal
import subprocess
import time
import json
from bench_lib import (
    MODEL_AWQ,
    hf_cache, results_vol,
    make_vllm_image,
    run_benchmark_impl, wait_for_server,
)

GPU = "A100"
CONCURRENCIES = [1, 64]

# Marlin args: drop --enforce-eager, use awq_marlin quantization
VLLM_MARLIN_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_AWQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "64",
    "--quantization", "awq_marlin",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--disable-log-requests",
]

app = modal.App("inference-bench-a100-marlin")


@app.cls(
    image=make_vllm_image(),
    gpu=GPU,
    timeout=60 * 20,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class MarlinTest:
    @modal.enter()
    def start(self):
        self.proc = subprocess.Popen(VLLM_MARLIN_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "Marlin server failed to start"
        self.cold = time.perf_counter() - t0

    @modal.method()
    def run_all(self) -> list[dict]:
        results = []
        for c in CONCURRENCIES:
            r = run_benchmark_impl("vllm", "short", c, repeat=1, model_name=MODEL_AWQ)
            r["cold_start_seconds"] = round(self.cold, 1)
            r["config"] = "awq_marlin"
            results.append(r)
        return results

    @modal.exit()
    def stop(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main():
    print("=" * 50)
    print("  A100 AWQ-Marlin test")
    print(f"  Model: {MODEL_AWQ}")
    print(f"  Concurrencies: {CONCURRENCIES}")
    print("=" * 50)

    bench = MarlinTest()
    results = bench.run_all.remote()

    for r in results:
        print(f"  c={r['concurrency']} | throughput={r.get('throughput_tokens_per_sec', 0):.0f} tok/s | ttft={r.get('ttft_p50', 0):.3f}s | config={r['config']}")

    outpath = "results/a100_marlin_results.jsonl"
    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n  Done. {len(results)} results → {outpath}")

"""A100 AWQ-Marlin c=16 midpoint — fills the one missing data point for curve fitting.

Usage: modal run modal_a100_marlin_c16.py
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

VLLM_MARLIN_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_AWQ,
    "--host", "0.0.0.0", "--port", "8000",
    "--max-num-seqs", "64",
    "--quantization", "awq_marlin",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--disable-log-requests",
]

app = modal.App("inference-bench-a100-marlin-c16")

@app.cls(
    image=make_vllm_image(),
    gpu=GPU, timeout=60 * 15, scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class MarlinC16:
    @modal.enter()
    def start(self):
        self.proc = subprocess.Popen(VLLM_MARLIN_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "Marlin server failed"
        self.cold = time.perf_counter() - t0

    @modal.method()
    def run(self) -> dict:
        r = run_benchmark_impl("vllm", "short", 16, repeat=1, model_name=MODEL_AWQ)
        r["cold_start_seconds"] = round(self.cold, 1)
        r["config"] = "awq_marlin"
        return r

    @modal.exit()
    def stop(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main():
    print("A100 AWQ-Marlin c=16 midpoint")
    bench = MarlinC16()
    r = bench.run.remote()
    print(f"  c=16 | throughput={r.get('throughput_tokens_per_sec', 0):.0f} tok/s | ttft={r.get('ttft_p50', 0):.3f}s")
    with open("results/a100_marlin_c16.jsonl", "w") as f:
        f.write(json.dumps(r) + "\n")
    print("  Done → results/a100_marlin_c16.jsonl")

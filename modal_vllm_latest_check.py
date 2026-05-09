"""T4.1 — vLLM version spot-check: dense 7B FP16 on A100 with latest vLLM.

Reruns the same config as the v0.8.5 A100 baseline to detect throughput drift.
If any concurrency shows >10% change, update the dashboard and README.

Baseline (vLLM v0.8.5, A100 80GB, Qwen2.5-7B FP16):
  c=1:  72.1 tok/s  | TTFT p50 0.024s | lat p95 1.96s
  c=16: 950.1 tok/s | TTFT p50 0.094s | lat p95 2.00s
  c=64: 2563.7 tok/s | TTFT p50 0.410s | lat p95 2.68s

Usage:  modal run modal_vllm_latest_check.py
Output: results/a100_vllm_latest_check.jsonl  (~$0.30-0.50, ~20 min)
"""
import json
import os
import subprocess
import time

import modal

from bench_lib import (
    MODEL_FP16,
    hf_cache,
    results_vol,
    run_benchmark_impl,
    wait_for_server,
)

GPU = "A100-80GB"
CONCURRENCIES = [1, 16, 64]
REGIME = "short"
VLLM_IMAGE = "vllm/vllm-openai:latest"
RESULT_FILE = "results/a100_vllm_latest_check.jsonl"

BASELINE = {
    1:  {"throughput": 72.1,   "ttft_p50": 0.024, "latency_p95": 1.96},
    16: {"throughput": 950.1,  "ttft_p50": 0.094, "latency_p95": 2.00},
    64: {"throughput": 2563.7, "ttft_p50": 0.410, "latency_p95": 2.68},
}

app = modal.App("inference-bench-vllm-latest-check")


def _make_image():
    _workload = os.path.join(os.path.dirname(__file__), "prompts", "workload.jsonl")
    _symlink = [
        "RUN for p in /usr/local/bin/python3 /usr/bin/python3 /opt/conda/bin/python3; do "
        'if [ -f "$p" ]; then ln -sf "$p" /usr/local/bin/python && break; fi; done '
        "&& python --version"
    ]
    return (
        modal.Image.from_registry(VLLM_IMAGE, setup_dockerfile_commands=_symlink)
        .entrypoint([])
        .pip_install("httpx>=0.27", "numpy>=1.26")
        .env({"HF_HOME": "/hf_cache"})
        .add_local_python_source("bench_lib")
        .add_local_file(_workload, remote_path="/opt/prompts/workload.jsonl")
    )


@app.cls(
    image=_make_image(),
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class LatestVLLMCheck:

    @modal.enter()
    def start_server(self):
        args = [
            "vllm", "serve", MODEL_FP16,
            "--host", "0.0.0.0", "--port", "8000",
            "--max-num-seqs", "64",
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.90",
            "--no-enable-log-requests",
        ]
        print(f"[server] Starting latest vLLM: {' '.join(args[:4])}")
        self.proc = subprocess.Popen(args)
        t0 = time.perf_counter()
        if not wait_for_server(timeout=300):
            self.proc.terminate()
            self.proc.wait(timeout=10)
            raise RuntimeError("Server failed to start")
        self.cold_start = time.perf_counter() - t0
        # Capture vLLM version for the result record
        try:
            v = subprocess.check_output(["vllm", "--version"], text=True).strip()
            self.vllm_version = v.split()[-1] if v else "unknown"
        except Exception:
            self.vllm_version = "unknown"
        print(f"[server] Ready in {self.cold_start:.1f}s  (vLLM {self.vllm_version})")

    @modal.method()
    def run_all(self) -> list[dict]:
        results = []
        for c in CONCURRENCIES:
            print(f"\n  >> FP16 c={c}")
            r = run_benchmark_impl("vllm", REGIME, c, repeat=1)
            r["cold_start_seconds"] = round(self.cold_start, 1)
            r["vllm_version"] = self.vllm_version
            r["config"] = "fp16_latest"
            r["gpu"] = GPU
            results.append(r)
            tp = r.get("throughput_tokens_per_sec", 0)
            ttft = r.get("ttft_p50", "N/A")
            lat = r.get("latency_p95", "N/A")
            print(f"     throughput={tp:.1f} tok/s  ttft_p50={ttft}  lat_p95={lat}")
        return results

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=15)


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  T4.1 — vLLM latest spot-check")
    print(f"  Model:  {MODEL_FP16}")
    print(f"  GPU:    {GPU}")
    print(f"  Image:  {VLLM_IMAGE}")
    print(f"  Conc:   {CONCURRENCIES}")
    print("=" * 60)

    bench = LatestVLLMCheck()
    results = bench.run_all.remote()

    # Save
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved {len(results)} results → {RESULT_FILE}")

    # Comparison table
    vllm_ver = results[0].get("vllm_version", "latest") if results else "latest"
    print(f"\n  {'':4} {'Baseline v0.8.5':>18} {'Latest ' + vllm_ver:>18} {'Delta':>8}")
    print(f"  {'-'*52}")
    any_material = False
    for r in results:
        c = r["concurrency"]
        new_tp = r.get("throughput_tokens_per_sec", 0)
        base_tp = BASELINE[c]["throughput"]
        delta = (new_tp - base_tp) / base_tp * 100
        flag = "  *** >10% CHANGE ***" if abs(delta) > 10 else ""
        if abs(delta) > 10:
            any_material = True
        print(f"  c={c:<3} {base_tp:>15.1f} tok/s  {new_tp:>15.1f} tok/s  {delta:>+6.1f}%{flag}")

    print()
    if any_material:
        print("  ACTION NEEDED: >10% throughput change detected.")
        print("  Update results/summary.jsonl, dashboard, and README with new numbers.")
    else:
        print("  All deltas within 10% — baseline numbers remain current.")

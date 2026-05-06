"""MoE benchmark — Qwen3-30B-A3B AWQ (Marlin) on A100 80GB, LONG CONTEXT.

Uses latest vLLM for MoE kernel optimizations (different from v0.8.5
used in dense model benchmarks).

VRAM: AWQ Int4 ~17 GB weights, but KV cache at max_model_len=16384
needs up to ~41 GB at c=16.
Concurrency: c=1, c=4, c=16.
Context length: 16384 (long context pressure test).

Usage: modal run modal_qwen3_moe_awq_long.py
"""
import modal
import subprocess
import time
import json
import os

from bench_lib import (
    hf_cache, results_vol,
    run_benchmark_impl, wait_for_server,
)

# ── Config ──
GPU = "A100"
TIMEOUT = 60 * 45  # 45 min

MODEL = "cognitivecomputations/Qwen3-30B-A3B-AWQ"  # AWQ Int4, ~17 GB
CONCURRENCIES = [1, 4, 16]
REGIME = "short"
MAX_MODEL_LEN = 16384  # long context
VLLM_IMAGE = "vllm/vllm-openai:v0.20.1"

app = modal.App("inference-bench-moe-awq-long")


def make_vllm_latest_image():
    """Build vLLM v0.20.1 image."""
    _workload = os.path.join(os.path.dirname(__file__), "prompts", "workload.jsonl")
    _symlink = [
        "RUN for p in /usr/local/bin/python3 /usr/bin/python3 /opt/conda/bin/python3; do "
        'if [ -f "$p" ]; then ln -sf "$p" /usr/local/bin/python && break; fi; done '
        "&& python --version"
    ]
    return (
        modal.Image.from_registry(
            VLLM_IMAGE,
            setup_dockerfile_commands=_symlink,
        )
        .entrypoint([])
        .pip_install("httpx>=0.27", "numpy>=1.26")
        .env({"HF_HOME": "/hf_cache"})
        .add_local_python_source("bench_lib")
        .add_local_file(_workload, remote_path="/opt/prompts/workload.jsonl")
    )


def _make_server_args(model: str, max_model_len: int = 4096, quantization: str | None = None) -> list[str]:
    """Build vLLM v0.20.1+ server args — uses 'vllm serve' CLI (model is positional)."""
    args = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-num-seqs", "16",
        "--max-model-len", str(max_model_len),
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.90",
        "--no-enable-log-requests",
    ]
    if quantization:
        args.extend(["--quantization", quantization])
    return args


@app.cls(
    image=make_vllm_latest_image(),
    gpu=GPU,
    timeout=TIMEOUT,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class MoEBenchmark:

    @modal.enter()
    def start_server(self):
        """Start vLLM server, wait for readiness."""
        args = _make_server_args(MODEL, max_model_len=MAX_MODEL_LEN, quantization="awq_marlin")
        print(f"[server] Starting: {' '.join(args)}")
        self.proc = subprocess.Popen(args)
        t0 = time.perf_counter()
        ok = wait_for_server(timeout=600)
        self.cold_start = time.perf_counter() - t0
        if not ok:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            raise RuntimeError(f"Server failed to start: {' '.join(args[:6])}")
        print(f"[server] Ready in {self.cold_start:.1f}s")

    @modal.method()
    def run(self, concurrency: int) -> dict:
        """Run a single benchmark at given concurrency."""
        config_label = "moe_awq_long"
        print(f"[bench] {config_label} c={concurrency} ...")
        r = run_benchmark_impl(
            engine="qwen3_moe_vllm",
            regime=REGIME,
            concurrency=concurrency,
            repeat=1,
            model_name=MODEL,
        )
        r["cold_start_seconds"] = round(self.cold_start, 1)
        r["config"] = config_label
        r["vllm_version"] = "v0.20.1"
        r["gpu"] = GPU
        r["model"] = MODEL
        print(f"[bench] {config_label} c={concurrency}: {r.get('throughput_tokens_per_sec', 'ERR')} tok/s, "
              f"success={r.get('successful_requests', '?')}/{r.get('total_requests', '?')}")
        return r

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=15)


@app.local_entrypoint()
def main():
    """Run AWQ MoE benchmark on A100 (long context, max_model_len=16384)."""
    print("=" * 60)
    print("  MoE AWQ Long-Context Benchmark: Qwen3-30B-A3B-AWQ on A100 80GB")
    print(f"  Model: {MODEL}")
    print(f"  Concurrencies: {CONCURRENCIES}")
    print(f"  Max model len: {MAX_MODEL_LEN} (long context)")
    print("=" * 60)

    bench = MoEBenchmark()
    all_results = []
    for c in CONCURRENCIES:
        r = bench.run.remote(concurrency=c)
        all_results.append(r)

    if all_results:
        results_path = "/results/moe_awq_16384_results.jsonl"
        with open(results_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        results_vol.commit()
        print(f"\nSaved {len(all_results)} results to {results_path}")

        print("\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        for r in all_results:
            label = f"{r.get('config', '?'):>15s} c={r.get('concurrency', '?'):>2d}"
            tp = r.get('throughput_tokens_per_sec', 'ERR')
            sr = f"{r.get('successful_requests', '?')}/{r.get('total_requests', '?')}"
            print(f"  {label}: {tp:>8.1f} tok/s  success={sr}")
    else:
        print("\nNo results — run failed.")
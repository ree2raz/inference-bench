"""MoE benchmark — Qwen3-30B-A3B BF16 on A100 80GB via vLLM v0.20.1.

Uses latest vLLM for MoE kernel optimizations (different from v0.8.5
used in dense model benchmarks).

VRAM: BF16 ~61 GB weights on A100 80GB.
Concurrency: c=1 and c=4 only (tight VRAM at higher concurrency).

Usage: modal run modal_qwen3_moe_bf16.py
"""
import modal
import subprocess
import time
import json
import os

from bench_lib import (
    hf_cache, results_vol,
    run_benchmark_impl, wait_for_server,
    print_result_line, write_summary,
)

# ── Config ──
GPU = "A100-80GB"
TIMEOUT = 60 * 180  # 3 hours

MODEL = "Qwen/Qwen3-30B-A3B"  # 30.5B total, 3.3B active, BF16, ~61 GB
CONCURRENCIES = [1, 4]
REGIME = "short"
VLLM_IMAGE = "vllm/vllm-openai:v0.20.1"

app = modal.App("inference-bench-moe-bf16")


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
        args = _make_server_args(MODEL, max_model_len=4096)
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
    def run_all(self):
        """Run all benchmark configurations."""
        engine = "qwen3_moe_bf16"
        all_results = []
        skipped = 0
        for conc in CONCURRENCIES:
            for rep in range(1, 2):  # repeat = 1
                run_id = f"{engine}_{REGIME}_c{conc}_r{rep}"
                remote_path = f"/results/raw/{run_id}.jsonl"
                if os.path.exists(remote_path):
                    print(f"  [skip] {run_id} already in Volume")
                    skipped += 1
                    continue
                print(f"\n  >> {engine} | {REGIME} | c={conc} | r={rep}/1")
                result = run_benchmark_impl(engine, REGIME, conc, rep, model_name=MODEL)
                result["cold_start_seconds"] = round(self.cold_start, 1)
                result["config"] = "moe_bf16"
                result["vllm_version"] = "v0.20.1"
                result["gpu"] = GPU
                result["model"] = MODEL
                print_result_line(result)
                all_results.append(result)
        if all_results:
            write_summary(all_results)
        if skipped:
            print(f"\n  Skipped {skipped} already-completed configs.")
        return len(all_results), skipped

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=15)


@app.local_entrypoint()
def main():
    """Run BF16 MoE benchmark on A100."""
    print("=" * 60)
    print("  MoE BF16 Benchmark: Qwen3-30B-A3B on A100 80GB")
    print(f"  Model: {MODEL}")
    print(f"  Concurrencies: {CONCURRENCIES}")
    print("=" * 60)

    bench = MoEBenchmark()
    n_new, n_skip = bench.run_all.remote()
    print(f"\n  Done: {n_new} new results, {n_skip} skipped.")
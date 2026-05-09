"""v2 sweep: vLLM v0.20.1 dense FP16 on A100 40GB.

Engine name: vllm_v2_a100. Concurrencies [1, 16, 64], REPEATS=1 (matches existing A100 pattern).
Usage: modal run --detach modal_v2_a100_vllm_fp16.py
"""
import modal
import os
import subprocess
import time
import json

from bench_lib import (
    MODEL_FP16,
    hf_cache, results_vol,
    make_vllm_v2_image,
    wait_for_server, VLLM_V2_SERVER_ARGS,
    run_benchmark_impl, print_result_line,
)

GPU = "A100"  # 40GB variant on Modal
CONCURRENCIES = [1, 16, 64]
ENGINE = "vllm_v2_a100"
app = modal.App("inference-bench-v2-a100-vllm-fp16")


@app.cls(
    image=make_vllm_v2_image(),
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class V2VllmA100Bench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(VLLM_V2_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "vLLM server failed to start"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] vLLM v0.20.1 A100 FP16 ready in {self.cold_start:.1f}s")

    @modal.method()
    def run_all(self) -> list[dict]:
        out = []
        for c in CONCURRENCIES:
            r = run_benchmark_impl(ENGINE, "short", c, repeat=1, model_name=MODEL_FP16)
            r["cold_start_seconds"] = round(self.cold_start, 1)
            r["engine_version"] = "vllm-v0.20.1"
            r["gpu"] = "A100-40GB"
            out.append(r)
        return out

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main():
    print(f"=== v2 sweep: {ENGINE} A100 40GB FP16 ===")
    bench = V2VllmA100Bench()
    results = bench.run_all.remote()
    for r in results:
        print_result_line(r)
    outpath = os.path.join(os.path.dirname(__file__), "results", "v2_a100_vllm_fp16.jsonl")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  {len(results)} configs -> {outpath}")

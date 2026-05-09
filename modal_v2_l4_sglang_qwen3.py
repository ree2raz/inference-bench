"""v2 sweep: SGLang :latest Qwen3-8B reasoning on L4.

Usage: modal run --detach modal_v2_l4_sglang_qwen3.py
"""
import modal
import os
import subprocess
import time
import json

from bench_lib import (
    GPU_TYPE, MODEL_QWEN3, REASONING_CONCURRENCY_LEVELS, REASONING_REPEATS,
    hf_cache, results_vol,
    make_sglang_v2_image,
    wait_for_server, QWEN3_SGLANG_SERVER_ARGS,
    run_benchmark_impl, print_result_line,
)

ENGINE = "sglang_v2_qwen3"
app = modal.App("inference-bench-v2-l4-sglang-qwen3")


@app.cls(
    image=make_sglang_v2_image(),
    gpu=GPU_TYPE,
    timeout=60 * 180,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class V2SglangL4Qwen3Bench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(QWEN3_SGLANG_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=600), "SGLang Qwen3 server failed to start"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] SGLang :latest Qwen3-8B ready in {self.cold_start:.1f}s")

    @modal.method()
    def run(self, concurrency: int, repeat: int) -> dict:
        result = run_benchmark_impl(ENGINE, "reasoning", concurrency, repeat, model_name=MODEL_QWEN3)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        result["engine_version"] = "sglang-latest"
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main():
    print(f"=== v2 sweep: {ENGINE} L4 Qwen3-8B reasoning ===")
    bench = V2SglangL4Qwen3Bench()
    all_results = []
    for conc in REASONING_CONCURRENCY_LEVELS:
        for rep in range(1, REASONING_REPEATS + 1):
            print(f"\n  >> {ENGINE} | reasoning | c={conc} | r={rep}/{REASONING_REPEATS}")
            r = bench.run.remote(concurrency=conc, repeat=rep)
            print_result_line(r)
            all_results.append(r)
    if all_results:
        outpath = os.path.join(os.path.dirname(__file__), "results", "v2_l4_sglang_qwen3.jsonl")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  {len(all_results)} configs -> {outpath}")
